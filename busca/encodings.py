import torch
from torch import nn 
import numpy as np
from positional_encodings.torch_encodings import PositionalEncoding3D  # https://github.com/tatp22/multidim-positional-encoding

from busca.tracking import missing_candidate_bbox


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, input_flavour, dropout=0.1, max_temp_dist=30, max_distance_dist=105, max_size_dist=105, encode_sep_as_ref=True, batch_first=False, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.input_flavour = input_flavour
        self.encode_sep_as_ref = encode_sep_as_ref
        self.device = device
        
        self.max_distance_dist = max_distance_dist
        self.max_size_dist = max_size_dist
        self.max_temp_dist = max_temp_dist
        self.distant_fake_bbox = torch.from_numpy(missing_candidate_bbox(flavour='ltwh')).to(self.device)

        distance_range = self.max_distance_dist * 2 + 1  
        size_range = self.max_size_dist * 2 + 1  # Smaller and bigger
        temp_range = self.max_temp_dist * 2 + 1  # Previous and forward frames
        

        p_enc_3d = PositionalEncoding3D(channels=self.d_model)  # Will receive a 5d tensor of size (batch_size, x, y, z, ch)
        pe = p_enc_3d(torch.zeros(1, distance_range, size_range, temp_range, d_model))  # Will return a 5d tensor of size (batch_size, x, y, z, ch)
        pe = pe.squeeze(0)  # Remove the batch dimension
        pe = pe.to(torch.float16)  # We use float16 to save memory
        self.pe = pe

        # We remove unnecerary stuff that may cause memory leaks
        del p_enc_3d.cached_penc
        del p_enc_3d.inv_freq
        
        assert self.batch_first == True, "batch_first must be True"
        
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, mem, can, mem_bboxes, can_bboxes, num_candidates):
        batch_size = mem.shape[0]

        ref_bbox = mem_bboxes[:, -1:, :].clone()  # [batch_size, 4]  | Our reference bbox is the last one in the memory

        if not self.batch_first:
            raise NotImplementedError("Not implemented for batch_first=False")

        # We first add "fake" bbox coordinates to the candidates extra tokens (SEP) and (ZERO)
        can_bboxes = self._insert_fake_bboxes(can=can, can_bboxes=can_bboxes, ref_bbox=ref_bbox, num_candidates=num_candidates, encode_sep_as_ref=self.encode_sep_as_ref)
        
        # We also need a fake bbox for the CLS token in the memory (if we are using it)
        if self.input_flavour.startswith('CLS-'):
            assert mem_bboxes.shape[1] == mem.shape[1] - 1, "The number of bboxes in the mem is not correct"
            cls_fake_bbox = ref_bbox
            mem_bboxes = torch.cat([cls_fake_bbox, mem_bboxes], dim=1)

        # Get the temporal indices for the embeddings
        mem_t_inds, can_t_inds = self._get_temporal_ids(mem=mem, can=can, num_candidates=num_candidates)

        # Get the spatial indices for the embeddings
        [mem_xy_inds, mem_size_inds], [can_xy_inds, can_size_inds] = self._get_spatial_ids(mem_bboxes=mem_bboxes, can_bboxes=can_bboxes)

        # Retrieve the encodings for the memory and candidates
        mem_indices = zip(mem_xy_inds, mem_size_inds, mem_t_inds)
        mem_encodings = []
        for sample_ind in mem_indices:  # We iterate batch_size elems
            elem_mem_indices = zip(sample_ind[0], sample_ind[1], sample_ind[2])  # i[0], i[1], i[2]
            elem_mem_encodings = torch.stack([self.pe[i[0].item(), i[1].item(), i[2].item()] for i in elem_mem_indices])  # Size [mem_len, d_model]
            mem_encodings.append(elem_mem_encodings)
            
        mem_encodings = torch.stack(mem_encodings)  # Size [batch_size, mem_len, d_model]
        mem_encodings = mem_encodings.to(self.device)

        can_indices = zip(can_xy_inds, can_size_inds, can_t_inds)
        can_encodings = []
        for sample_ind in can_indices:  # We iterate batch_size elems
            elem_can_indices = zip(sample_ind[0], sample_ind[1], sample_ind[2])  # i[0], i[1], i[2]
            elem_can_encodings = torch.stack([self.pe[i[0].item(), i[1].item(), i[2].item()] for i in elem_can_indices])  # Size [num_cans, d_model]
            can_encodings.append(elem_can_encodings)
            
        can_encodings = torch.stack(can_encodings)  # Size [batch_size, num_cans, d_model]
        can_encodings = can_encodings.to(self.device)
        
        mem = mem + mem_encodings
        can = can + can_encodings

        # Now we concat the two
        concat_dim = 1 if self.batch_first else 0
        x = torch.cat((mem, can), dim=concat_dim)

        return self.dropout(x)
    

    def _insert_fake_bboxes(self, can, can_bboxes, ref_bbox, num_candidates, encode_sep_as_ref=True):
        """We need to insert fake bboxes for the [SEP] tokens and the [ZERO] token (as they don't have bboxes)
        The bboxes for all of them will be the last bbox in the memory (ref_bbox). Unless encode_sep_as_ref=False,
        in which case the [SEP] tokens will have the bbox of the [CAN] tokens (except for [ZERO] and [BAD], which will
        be encoded with ref_bbox).
        num_candidates is the number of candidates in the input (including [BAD] and [ZERO] if present)"""
        batch_size = can.shape[0]

        if 'BAD' in self.input_flavour:  # We are separating the MISSING and BAD_MEMORY candidates
            num_additional_candidates = 2
            distant_fake_bboxes = self.distant_fake_bbox.repeat(batch_size, 1, 1)
        else:
            num_additional_candidates = 1
            distant_fake_bboxes = None
        
        assert can.shape[1] == (can_bboxes.shape[1] + num_additional_candidates) * 2, "The number of tokens in the candidates is not correct"
        assert can_bboxes.shape[1] == num_candidates - num_additional_candidates, "The number of bboxes in the candidates is not correct"

        # We need to insert fake bboxes for the [SEP] tokens and the [ZERO] token (as they don't have bboxes) (and maybe [BAD] too)
        # The bboxes for all of them will be the last bbox in the memory (ref_bbox)
        if self.input_flavour in {'CLS-MEM-SEP-CAN', 'MEM-SEP-CAN', 'CLS-MEM-SEP-CAN-BAD', 'MEM-SEP-CAN-BAD'}:
            if encode_sep_as_ref:
                can_with_pads = [torch.cat([ref_bbox, can_bboxes[:, [i], :]], dim=1) for i in range(can_bboxes.shape[1])]
            else:
                can_with_pads = [torch.cat([can_bboxes[:, [i], :], can_bboxes[:, [i], :]], dim=1) for i in range(can_bboxes.shape[1])]
            
            zero_bboxes = [ref_bbox, ref_bbox]
            bad_bboxes = [distant_fake_bboxes, distant_fake_bboxes]  # We don't wanna use the last bbox in the memory for the BAD_MEMORY candidates
            
            if 'BAD' in self.input_flavour:
                fake_bboxes = torch.cat(can_with_pads + zero_bboxes + bad_bboxes, dim=1)  # torch.Size([128, 14, 512]) # 128 is batch size, 14 is 5 candidates + ZERO + BAD + seps, 512 is d_model
            else:
                fake_bboxes = torch.cat(can_with_pads + zero_bboxes, dim=1)  # torch.Size([128, 12, 512]) # 128 is batch size, 12 is 5 candidates + ZERO + seps, 512 is d_model
        
        elif self.input_flavour in {'CLS-MEM-CAN-SEP', 'MEM-CAN-SEP', 'CLS-MEM-CAN-SEP-BAD', 'MEM-CAN-SEP-BAD'}:
            if encode_sep_as_ref:
                can_with_pads = [torch.cat([can_bboxes[:, [i], :], ref_bbox], dim=1) for i in range(can_bboxes.shape[1])]
            else:
                can_with_pads = [torch.cat([can_bboxes[:, [i], :], can_bboxes[:, [i], :]], dim=1) for i in range(can_bboxes.shape[1])]
            
            zero_bboxes = [ref_bbox, ref_bbox]
            bad_bboxes = [distant_fake_bboxes, distant_fake_bboxes]  # We don't wanna use the last bbox in the memory for the BAD_MEMORY candidates
            
            if 'BAD' in self.input_flavour:
                fake_bboxes = torch.cat(can_with_pads + zero_bboxes + bad_bboxes, dim=1)  # torch.Size([128, 14, 512]) # 128 is batch size, 14 is 5 candidates + ZERO + BAD + seps, 512 is d_model
            else:
                fake_bboxes = torch.cat(can_with_pads + zero_bboxes, dim=1)  # torch.Size([128, 12, 512]) # 128 is batch size, 12 is 5 candidates + ZERO + seps, 512 is d_model
        
        else:
            raise ValueError(f"Unknown input_flavour: {self.input_flavour}")

        return fake_bboxes
    
    def _get_temporal_ids(self, mem, can, num_candidates, range_factor=2.0):
        """num_candidates is the number of candidates in the input (including [BAD] and [ZERO] if present)"""
        if not self.batch_first:
            raise NotImplementedError("Not implemented for batch_first=False")
        
        batch_size = mem.shape[0]
        
        # We want for mem[-1] to be 0, the previous memory to be negative and for the candidates to be positive
        mem_len = mem.shape[1]
        mem_inds = torch.tensor(list(range(-mem_len+1, 0+1)))  # +1 because range is not inclusive
        if self.input_flavour.startswith('CLS-'):
            mem_inds = mem_inds[0] = 0  # We want the CLS token to have temporal id 0

        can_len = can.shape[1]
        assert can_len % num_candidates == 0, "The number of tokens in the candidates is not correct"
        elems_per_can = int(can_len / num_candidates)  # "[SEP] [CAN1] [SEP] [CAN2] [SEP] [CAN3] [SEP] [CAN4] [SEP] [CAN5] [SEP] [ZERO]"  -> This will be 12 / (5+1) = 2 elems per can
        can_inds = torch.tensor(list(range(1, 1 + elems_per_can)) * num_candidates)

        # We now apply the range factor
        mem_inds = torch.clamp(mem_inds * range_factor, min=-self.max_temp_dist, max=self.max_temp_dist).to(torch.long)
        can_inds = torch.clamp(can_inds * range_factor, min=-self.max_temp_dist, max=self.max_temp_dist).to(torch.long)

        # We now shift the indices, so that what now is 0 lies in the middle of the range
        mem_inds = mem_inds + self.max_temp_dist
        can_inds = can_inds + self.max_temp_dist

        # We now repeat the indices to the batch size
        mem_inds = mem_inds.repeat(batch_size, 1)
        can_inds = can_inds.repeat(batch_size, 1)

        return mem_inds, can_inds
    
    
    def _get_spatial_ids(self, mem_bboxes, can_bboxes, range_factor=15.0):
        if not self.batch_first:
            raise NotImplementedError("Not implemented for batch_first=False")
        
        batch_size = mem_bboxes.shape[0]
        
        ref_bbox = mem_bboxes[:, -1:, :]  # [batch_size, 4]  | Our reference bbox is the last one in the memory
        
        # We will replicate the reference bbox to the same size as the can_bboxes and the whole memory
        ref_bbox_can = ref_bbox.repeat(1, can_bboxes.shape[1], 1)  # [batch_size, num_candidates, 4]
        ref_bbox_mem = ref_bbox.repeat(1, mem_bboxes.shape[1], 1)  # [batch_size, mem_size, 4]
        
        # Get the spatial indices for the embeddings
        # We will flatten the bboxes along the batch dimension
        # can_bboxes [batch_size, num_candidates, 4] -> [batch_size * num_candidates, 4]
        can_bboxes_flattened = can_bboxes.view(-1, 4)
        ref_bbox_can_flattened = ref_bbox_can.view(-1, 4)

        # mem_bboxes [batch_size, mem_size, 4] -> [batch_size * mem_size, 4]
        mem_bboxes_flattened = mem_bboxes.view(-1, 4)
        ref_bbox_mem_flattened = ref_bbox_mem.view(-1, 4)

        # We get the distances for the CANs
        can_xy_distance_flattened, can_size_distance_flattened = self.extract_distance_values(bbox=can_bboxes_flattened, ref_bbox=ref_bbox_can_flattened)
        can_xy_distance = can_xy_distance_flattened.view(batch_size, -1)
        can_size_distance = can_size_distance_flattened.view(batch_size, -1)

        # Values roughly go from -7 to 7
        # Much like MEGA does, we will multiply by a factor (15, instead of 100) to get a wider range and get values from -105 to 105
        can_xy_distance = torch.clamp(can_xy_distance * range_factor, min=-self.max_distance_dist, max=self.max_distance_dist).to(torch.long)
        can_size_distance = torch.clamp(can_size_distance * range_factor, min=-self.max_size_dist, max=self.max_size_dist).to(torch.long)
        can_size_distance = torch.clamp(can_size_distance, min=-self.max_size_dist, max=self.max_size_dist).to(torch.long)  # I have to do it 2 times to avoid area-zero bboxes that overflow the metric

        # We get the distances for the MEMs
        mem_xy_distance_flattened, mem_size_distance_flattened = self.extract_distance_values(bbox=mem_bboxes_flattened, ref_bbox=ref_bbox_mem_flattened)
        mem_xy_distance = mem_xy_distance_flattened.view(batch_size, -1)
        mem_size_distance = mem_size_distance_flattened.view(batch_size, -1) 

        # Values roughly go from -7 to 7
        # Much like MEGA does, we will multiply by a factor (15, instead of 100) to get a wider range and get values from -105 to 105
        mem_xy_distance = torch.clamp(mem_xy_distance * range_factor, min=-self.max_distance_dist, max=self.max_distance_dist).to(torch.long)
        mem_size_distance = torch.clamp(mem_size_distance * range_factor, min=-self.max_size_dist, max=self.max_size_dist).to(torch.long)

        # Now we set the origin values of the distances
        # size should be around the center of the matrix (makes sense to represent bigger and smaller). Same size is now 0, smaller is negative, bigger is positive
        # xy should start at the beginning of the matrix (we don't want to distinguish between left and right). Same position is now -105, (roughly) and more distant is a bigger number
        mem_xy_inds = mem_xy_distance + self.max_distance_dist
        mem_size_inds = mem_size_distance + self.max_size_dist

        can_xy_inds = can_xy_distance + self.max_distance_dist
        can_size_inds = can_size_distance + self.max_size_dist

        return [mem_xy_inds, mem_size_inds], [can_xy_inds, can_size_inds]
        

    @staticmethod
    def extract_distance_values(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.tensor_split(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.tensor_split(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref
        delta_x = delta_x / bbox_width
        delta_x = torch.pow(delta_x, 2)

        delta_y = center_y - center_y_ref
        delta_y = delta_y / bbox_height
        delta_y = torch.pow(delta_y, 2)

        # We will use the euclidean distance, instead of element-by-element distance
        xy_distance = torch.sqrt(delta_x + delta_y)
        xy_distance = (xy_distance + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref
        delta_width = (delta_width + 1e-3).log()

        delta_height = bbox_height / bbox_height_ref
        delta_height = (delta_height + 1e-3).log()

        size_distance = delta_width + delta_height

        return xy_distance, size_distance