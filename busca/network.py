import torch
from torch import nn 
import numpy as np

from busca.custom_layers import TransformerEncoder, TransformerEncoderLayer
from busca.encodings import PositionalEncoding
from busca.reid.load_trained_net import load_net
from busca.tracking import get_bbox_crop, missing_candidate_bbox


class BUSCA(nn.Module):
    def __init__(self, args):
        super(BUSCA, self).__init__()
        self.args = args

        self.dim_embedding = args.dim_embedding
        self.dim_model = args.trans_dim  # Number of channels for the internal vectors
        self.activation = self._get_activation_fn(self.args.activation)

        self.build()


    def _get_activation_fn(self, activation='relu'):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == 'silu':
            return nn.SiLU()
        raise RuntimeError("activation should be relu/gelu/tanh/silu, not {}".format(activation))
    

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def build(self):
        self.pos_encoder = PositionalEncoding(self.dim_model, input_flavour=self.args.input_flavour, dropout=self.args.dropout_p, encode_sep_as_ref=self.args.encode_separator_as_reference, batch_first=True, device=self.args.device)

        if self.args.encode_special_tokens:
            special_tokens_size = self.dim_embedding
        else:
            special_tokens_size = self.dim_model
        
        if 'CLS' in self.args.input_flavour:
            self.cls_token = torch.nn.Parameter(torch.randn(special_tokens_size), requires_grad=True)
        else:
            self.cls_token = None
        
        if 'SEP' in self.args.input_flavour:
            self.sep_token = torch.nn.Parameter(torch.randn(special_tokens_size), requires_grad=True)
        else:
            self.sep_token = None
        
        self.non_token = torch.nn.Parameter(torch.randn(special_tokens_size), requires_grad=True)

        if 'BAD' in self.args.input_flavour:
            self.bad_token = torch.nn.Parameter(torch.randn(special_tokens_size), requires_grad=True)
        else:
            self.bad_token = None

        self.pad_token = torch.zeros(special_tokens_size)

        self.expected_image_size = None

        # ReID encoder (feature extractor)
        # We first check if we have the argument num_training_ids
        if not hasattr(self.args, "num_training_ids"):
            reid_encoder_num_classes = 299  # Placeholder value
        else:
            reid_encoder_num_classes = self.args.num_training_ids

        self.reid_encoder = ReID_Encoder(num_classes=reid_encoder_num_classes, device=self.args.device, pretrained_path=self.args.reid_weights_file,
                                        use_domain_adaptation=True, trainable=False, use_checkpointing=False).to(self.args.device)
        
        self.expected_image_size = self.reid_encoder.PRETRAINED_SIZE

        # We use our custom layers in order to be able to retrieve the self-attention weights
        encoder_layers = TransformerEncoderLayer(d_model=self.dim_model, nhead=self.args.nhead, dim_feedforward=self.args.ff_size,
                                                 dropout=self.args.dropout_p, activation=self.activation, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.args.num_layer)

        self.encoder = nn.Linear(self.dim_embedding, self.dim_model)  # Instead of an embedding, we use a linear layer
        
        self.decoder = nn.Sequential(nn.LayerNorm(self.dim_model),
                                    nn.Linear(self.dim_model, 1))  # 1 output that will be later forwarded to a softmax (with the other candidates)

        self.softmax = nn.Softmax(dim=-1)

        self.attentions = None  # To store the self-attention weights
        self.logits = None  # To store the logits
        self.mem_logits = None  # To store the logits for the memory tokens
    
    
    def _assemble_input(self, embeddings_memory, candidate_embeddings):
        """We have 6 different types of tokens:
        - MEM: Memory tokens storing the past observations of a given track
        - CAN: Candidate tokens for each proposal (detection or Kalman prediction)
        - SEP: Separator token that divides the different candidates

        - NON: Learned token that represents that none of the candidates is a good match
        - BAD: Learned token that represents a bad memory (i.e. a track that is corrups)
        - CLS: Unused token
        """
        relevant_positions = {'CLS': None, 'CAN': None, 'MEM': None, 'BAD': None, 'NON': None}
        batch_size = embeddings_memory.shape[0]
        
        if self.args.input_flavour.startswith('CLS-'):
            cls_token = self.cls_token.repeat(batch_size, 1).unsqueeze(1)
            input_mem_embeddings = torch.cat([cls_token, embeddings_memory], dim=1)
            relevant_positions['CLS'] = 0
            relevant_positions['MEM'] = slice(1, int(embeddings_memory.shape[1]+1))

        else:
            input_mem_embeddings = embeddings_memory
            relevant_positions['CLS'] = None
            relevant_positions['MEM'] = slice(0, int(embeddings_memory.shape[1]))
        
        if 'BAD' in self.args.input_flavour:
            non_token = self.non_token.repeat(batch_size, 1).unsqueeze(1)  # We will always add the non-token
            bad_token = self.bad_token.repeat(batch_size, 1).unsqueeze(1)
            candidate_embeddings = torch.cat([candidate_embeddings, non_token, bad_token], dim=1)  # Add the non-token
            num_candidates = candidate_embeddings.shape[1]
        
        else:
            non_token = self.non_token.repeat(batch_size, 1).unsqueeze(1)  # We will always add the non-token
            candidate_embeddings = torch.cat([candidate_embeddings, non_token], dim=1)  # Add the non-token
            num_candidates = candidate_embeddings.shape[1]
        
        if 'MEM-SEP-CAN' in self.args.input_flavour:  # [|||MEM|||] [SEP] [CA1] [SEP] [CA2] [SEP] [CA3] [SEP] [CA0]
            sep_token = self.sep_token.repeat(batch_size, 1).unsqueeze(1)
            input_can_embeddings = torch.cat([torch.cat([sep_token, candidate_embeddings[:, [i], :]], dim=1) for i in range(num_candidates)], dim=1)
            candidates_start = input_mem_embeddings.shape[1]
            relevant_positions['CAN'] = tuple([candidates_start + i for i in range(1, num_candidates*2 + 1, 2)])  # 2 because we have a sep_token between each candidate (and we start at 1)
            if 'BAD' in self.args.input_flavour:
                relevant_positions['BAD'] = -1
                relevant_positions['NON'] = -3
            else:
                relevant_positions['BAD'] = None
                relevant_positions['NON'] = -1

        elif 'MEM-CAN-SEP' in self.args.input_flavour:  # [|||MEM|||] [CA1] [SEP] [CA2] [SEP] [CA3] [SEP] [CA0] [SEP]
            sep_token = self.sep_token.repeat(batch_size, 1).unsqueeze(1)
            input_can_embeddings = torch.cat([torch.cat([candidate_embeddings[:, [i], :], sep_token], dim=1) for i in range(num_candidates)], dim=1)
            candidates_start = input_mem_embeddings.shape[1]
            relevant_positions['CAN'] = tuple([candidates_start + i for i in range(0, num_candidates*2, 2)])  # 2 because we have a sep_token between each candidate (and we start at 1)
            if 'BAD' in self.args.input_flavour:
                relevant_positions['BAD'] = -2
                relevant_positions['NON'] = -4
            else:
                relevant_positions['BAD'] = None
                relevant_positions['NON'] = -2

        else:
            raise NotImplementedError('Input flavour "{}" not implemented'.format(self.args.input_flavour))
        
        return [input_mem_embeddings, input_can_embeddings], relevant_positions
    

    def _retrieve_output(self, output_embeddings, relevant_positions, output_flavour=None):
        if output_flavour is None:
            output_flavour = self.args.output_flavour

        relevant_output = output_embeddings[:, relevant_positions[output_flavour]]
        
        return relevant_output
    
    def forward(self, embeddings_memory, candidate_embedding, memory_bboxes=None, candidates_bboxes=None, return_att=False, return_logits=False, plot_results=False):
        num_candidates = candidate_embedding.shape[1]

        if plot_results:
            image_list_mem = embeddings_memory.permute(0, 1, 3, 4, 2).detach().cpu().numpy()[..., ::-1]  # From [C, H, W](RGB) to [H, W, C](BGR)
            image_list_can = candidate_embedding.permute(0, 1, 3, 4, 2).detach().cpu().numpy()[..., ::-1]  # From [C, H, W](RGB) to [H, W, C](BGR)


        # ReID encoder (feature extractor)
        # We will "collapse" the batch dimension and the mem_len/num_cans dimension
        re_batch_size, re_mem_len, re_mem_c, re_mem_h, re_mem_w = embeddings_memory.shape
        re_batch_size, re_can_len, re_can_c, re_can_h, re_can_w  = candidate_embedding.shape
        embeddings_memory = embeddings_memory.view(-1, re_mem_c, re_mem_h, re_mem_w)
        candidate_embedding = candidate_embedding.view(-1, re_can_c, re_can_h, re_can_w)

        # ReID expects CHW
        mem_reid_cls, embeddings_memory_feat = self.reid_encoder(embeddings_memory)
        can_reid_cls, candidate_embedding_feat = self.reid_encoder(candidate_embedding)

        # We will "uncollapse" the batch dimension and the mem_len/num_cans dimension
        embeddings_memory_feat = embeddings_memory_feat.view(re_batch_size, re_mem_len, -1)  # Now shape is (batch_size, mem_len, 2048)
        candidate_embedding_feat = candidate_embedding_feat.view(re_batch_size, re_can_len, -1)  # Now shape is (batch_size, can_len, 2048)

        embeddings_memory = embeddings_memory_feat
        candidate_embedding = candidate_embedding_feat


        embeddings_memory = self.encoder(embeddings_memory) * np.sqrt(self.dim_model)
        candidate_embedding = self.encoder(candidate_embedding) * np.sqrt(self.dim_model)

        input_seq, relevant_positions = self._assemble_input(embeddings_memory=embeddings_memory, candidate_embeddings=candidate_embedding)
        input_mem_seq, input_can_seq = input_seq
        
        if 'BAD' in self.args.input_flavour:
            total_num_candidates = num_candidates + 2  # +1 because we add the non-token +1 because we add the bad token
        else:
            total_num_candidates = num_candidates + 1 # +1 because we add the non-token
        input_seq = self.pos_encoder(mem=input_mem_seq, can=input_can_seq, mem_bboxes=memory_bboxes, can_bboxes=candidates_bboxes, num_candidates=total_num_candidates)

        transformer_output = self.transformer_encoder(input_seq, mask=None, src_key_padding_mask=None, return_att=return_att)
        if return_att:
            output_all, self.attentions = transformer_output
        
        else:
            output_all = transformer_output
        
        output = self._retrieve_output(output_all, relevant_positions=relevant_positions, output_flavour=self.args.output_flavour)

        if return_logits:
            self.logits = output
            self.mem_logits = self._retrieve_output(output_all, relevant_positions=relevant_positions, output_flavour='MEM')

            # We now perform global average pooling over the memory tokens
            self.mem_logits = self.mem_logits.mean(dim=1)

        output = self.decoder(output)  # torch.Size([B, num_can+1, 1]); or torch.Size([B, num_can+2, 1]) if we use the BAD token
        output = output[:, :, 0]  # torch.Size([B, num_can+1]); or torch.Size([B, num_can+2]) if we use the BAD token

        if plot_results:
            import cv2
            from busca.visualization import create_batch_image
            output_probs = self.softmax(output).detach().cpu().numpy()
            print('Showing images')
            batch_img = create_batch_image(image_list_mem=image_list_mem, image_list_can=image_list_can, output_probs=output_probs, max_batch_size=5)

            cv2.imshow('image', batch_img)
            cv2.waitKey(0)

        return output
    
    
    def _get_track_mem(self, track, seq_len, use_broader_memory):
        full_mem_list = track.images_mem

        if use_broader_memory:
            if seq_len == 1 and len(full_mem_list) >= 1:
                # If seq_len is 1, we will return the last element of the memory
                embedding_mem = full_mem_list[-seq_len:]
                embedding_bboxes = track.tlwh_mem[-seq_len:]

            elif len(full_mem_list) < seq_len:
                # We will return an incomplete memory employing the usual strategy
                embedding_mem = full_mem_list[-seq_len:]
                embedding_bboxes = track.tlwh_mem[-seq_len:]
            
            else:
                # We will sample the memory trying for it to be equally distributed, from first to last element
                # We will sample seq_len elements from the memory, and we do NOT want to repeat any element
                elem_sep = float(len(full_mem_list)-1) / float(seq_len-1)
                embedding_mem = []
                embedding_bboxes = []
                for i in range(seq_len):
                    embedding_mem.append(full_mem_list[int(i * elem_sep)])
                    embedding_bboxes.append(track.tlwh_mem[int(i * elem_sep)])
                
                assert len(embedding_mem) == seq_len

        else:
            embedding_mem = full_mem_list[-seq_len:]
            embedding_bboxes = track.tlwh_mem[-seq_len:]
        
        embedding_bboxes = np.array(embedding_bboxes) * track.scale  # tlwh_mem are the bboxes in the original image, not the scaled ones output by the detector. That's why we have to convet them to "detector" coordinates
        
        return embedding_mem, embedding_bboxes
 
 
    def associate_embeddings(self, tracks_embeddings, dets_embeddings, dists_matrix, seq_len, num_candidates, use_broader_memory, select_highest_candidate, highest_candidate_minimum_thresh=None, keep_highest_value=False, extra_kalman_candidates=[], plot_results=False, normalize_ims=False):
        # dists_matrix is a matrix with the center-center distances between each track and each det
        # extra_kalman_candidates is a list that indicates if we want to include the kalman prediction as a candidate (and the list contains the kalman prediction for each track)
        base_im_dtype = np.uint8 if normalize_ims else np.float32  # We assume that the images will be uint8 and after normalization we will have float32

        if len(tracks_embeddings) == 0:
            return None, None
        
        if len(dets_embeddings) == 0 and len(extra_kalman_candidates) == 0:
            return None, None

        # First we will build the memories for each track
        complete_embeddings = []  # We will keep track of the embeddings that are not complete
        tracks_embeddings_batch = []
        tracks_bboxes_batch = []
        for track in tracks_embeddings:
            embedding_mem, embedding_bboxes = self._get_track_mem(track=track, seq_len=seq_len, use_broader_memory=use_broader_memory)

            if len(embedding_mem) == seq_len:
                # Everything is correct
                complete_embeddings.append(1.0)
            else:
                # We don't have enough information
                # As we are using ReID, we will use the ReID embeddings (a.k.a. images)
                embedding_mem = np.zeros([seq_len, self.expected_image_size[0], self.expected_image_size[1], 3], dtype=base_im_dtype)
                embedding_bboxes = np.zeros([seq_len, 4]) + np.array([+250, +250, +500, +500])  # We add a value to avoid NaNs (the value is somewhat big to avoid problems with missing cans)
                complete_embeddings.append(0.0)

            tracks_embeddings_batch.append(embedding_mem)
            tracks_bboxes_batch.append(embedding_bboxes)
        
        tracks_embeddings_batch = np.array(tracks_embeddings_batch, dtype=base_im_dtype)
        if normalize_ims:
            tracks_embeddings_batch = self._normalize_embeddings_batch(tracks_embeddings_batch)
        tracks_embeddings_batch = torch.from_numpy(tracks_embeddings_batch).float().to(self.args.device)
        
        tracks_bboxes_batch = np.array(tracks_bboxes_batch)
        tracks_bboxes_batch = torch.from_numpy(tracks_bboxes_batch).float()

        complete_embeddings = np.array(complete_embeddings)
        complete_embeddings = torch.from_numpy(complete_embeddings).float().to(self.args.device)

        # Now we will build the candidates for each track
        dets_embeddings_batch = []
        dets_bboxes_batch = []
        dets_embeddings_inds = []
        num_available_candidates = min(len(dets_embeddings), num_candidates)  # Maybe we don't have enough detections
        for t_ind, track in enumerate(tracks_embeddings):
            # We will build the candidates for this track
            # We will pick the `num_candidates` closest detections
            track_dists = dists_matrix[t_ind]
            track_dets_inds = np.argsort(track_dists)[:num_candidates].tolist()
            if len(track_dets_inds) < num_candidates:
                # We don't have enough detections
                # we will pad with zeros
                missing_dets = num_candidates - len(track_dets_inds)
                track_dets_inds.extend([None] * missing_dets)

            dets_embeddings_inds.append(track_dets_inds)

            track_dets_embeddings = []
            track_dets_bboxes = []
            for d_ind in track_dets_inds:
                if d_ind is not None:
                    det = dets_embeddings[d_ind]
                    # As we are using ReID, we will use the ReID embeddings (a.k.a. images)
                    track_dets_embeddings.append(det.images_mem[-1])
                    det_bbox = det.tlwh_mem[-1] * det.scale
                    track_dets_bboxes.append(det_bbox)
                
                else:
                    # As we are using ReID, we will use the ReID embeddings (a.k.a. images)
                    track_dets_embeddings.append(np.zeros([self.expected_image_size[0], self.expected_image_size[1], 3], dtype=base_im_dtype))
                    track_dets_bboxes.append(missing_candidate_bbox(flavour='ltwh'))  # For missing dets, we append a very unrealistic bbox
            
            dets_embeddings_batch.append(track_dets_embeddings)
            dets_bboxes_batch.append(track_dets_bboxes)
        
        # If we want to include Kalman as a candidate, we will do it here
        # We will add a candidate for each track, which is the prediction of the kalman filter
        # This candidate will replace the last one
        if len(extra_kalman_candidates) > 0:
            num_available_candidates = min(len(dets_embeddings) + 1, num_candidates)  # Note the +1 because we are adding a candidate (the kalman candidate)

            for t_ind, track in enumerate(tracks_embeddings):
                # We will build the extra kalman candidate for this track
                new_ind = len(dets_embeddings) + t_ind # It will be appended after the detections
                new_det = extra_kalman_candidates[t_ind]
                new_bbox = new_det.tlwh * new_det.scale

                kalman_ind_in_batch = min(len(dets_embeddings), num_candidates - 1)  # It will be appended after the detections, unless there are more detections than candidates, then it will replace the last one

                assert self.reid_encoder is not None, "We need to use ReID to use the kalman filter (as we need to use the images because we don't have detection embeddings)"

                new_embedding = new_det.images_mem[-1]

                dets_embeddings_inds[t_ind][kalman_ind_in_batch] = new_ind  # We will replace the last one with the kalman prediction
                dets_bboxes_batch[t_ind][kalman_ind_in_batch] = new_bbox
                dets_embeddings_batch[t_ind][kalman_ind_in_batch] = new_embedding

        
        dets_embeddings_batch = np.array(dets_embeddings_batch, dtype=base_im_dtype)
        if normalize_ims:
            dets_embeddings_batch = self._normalize_embeddings_batch(dets_embeddings_batch)
        dets_embeddings_batch = torch.from_numpy(dets_embeddings_batch).float().to(self.args.device)

        dets_bboxes_batch = np.array(dets_bboxes_batch)
        dets_bboxes_batch = torch.from_numpy(dets_bboxes_batch).float()

        # Now we transform the bboxes to the format that the Transformer expects
        # This is, from ltwh to [xmin, ymin, xmax, ymax]
        tracks_bboxes_batch = self.ltwh_to_ltrb(tracks_bboxes_batch).to(self.args.device)
        dets_bboxes_batch = self.ltwh_to_ltrb(dets_bboxes_batch).to(self.args.device)

        # If we are using BoT, forward expects [C, H, W](RGB) images, so we convert from from [H, W, C](BGR) to [C, H, W](RGB)
        tracks_embeddings_batch = tracks_embeddings_batch[..., [2, 1, 0]].permute(0, 1, 4, 2, 3)  # From [H, W, C](BGR) to [C, H, W](RGB)
        dets_embeddings_batch = dets_embeddings_batch[..., [2, 1, 0]].permute(0, 1, 4, 2, 3)  # From [H, W, C](BGR) to [C, H, W](RGB)

        # Now we feed the Transformer
        outputs = self.forward(embeddings_memory=tracks_embeddings_batch, candidate_embedding=dets_embeddings_batch, memory_bboxes=tracks_bboxes_batch, candidates_bboxes=dets_bboxes_batch,
                               return_att=False, return_logits=True, plot_results=plot_results)
        output_probs = self.softmax(outputs)
        
        output_probs = output_probs.cpu().detach().numpy()
        
        # Now we will add the probs to the global matrix
        num_dets = len(dets_embeddings) if len(extra_kalman_candidates) == 0 else len(dets_embeddings) + len(extra_kalman_candidates)  # If we are using the kalman filter as a candidate, we will add one more candidate
        probs_matrix = np.zeros([len(tracks_embeddings), num_dets])
        for t_ind, track in enumerate(tracks_embeddings):
            track_dets_inds = dets_embeddings_inds[t_ind]
            track_dets_inds = track_dets_inds[:num_available_candidates]  # We only keep the first `num_candidates` probs. We don't want the ZERO (a.k.a missing) prob (or any other filler candidates)
            track_probs = output_probs[t_ind]

            if select_highest_candidate: # If we are using the highest candidate, we will only keep the highest candidate (rest will be set to 0)
                track_probs_new = np.zeros_like(track_probs)
                if highest_candidate_minimum_thresh is None or highest_candidate_minimum_thresh == 0 or (highest_candidate_minimum_thresh > 0.0 and np.max(track_probs) >= highest_candidate_minimum_thresh):
                    if keep_highest_value:
                        track_probs_new[np.argmax(track_probs)] = np.max(track_probs)
                    else:
                        track_probs_new[np.argmax(track_probs)] = 1.0
                track_probs = track_probs_new

            track_probs = track_probs[:num_available_candidates]  # We only keep the first `num_candidates` probs. We don't want the ZERO (a.k.a missing) prob (or any other filler candidates)
            probs_matrix[t_ind, track_dets_inds] = track_probs

        reliable_predictions = complete_embeddings.cpu().detach().numpy().astype(bool)  # We return this so we can later ignore some of the predictions

        return probs_matrix, reliable_predictions

    
    def load_pretrained(self, path, ignore_reid=False, ignore_reid_fc=False):
        if not torch.cuda.is_available():
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(path)
        
        if 'model_state_dict' in state_dict.keys():
            model_state_dict = state_dict['model_state_dict']
            optimizer_state_dict = state_dict['optimizer_state_dict']
        else:
            model_state_dict = state_dict
            optimizer_state_dict = None
        
        if ignore_reid_fc:
            # We remove the layers that we don't need
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'reid_encoder.model.fc.' not in k and 'reid_encoder.model.fc_person.' not in k}
        
        if ignore_reid:
            # We remove the layers that we don't need
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'reid_encoder.model.' not in k}
        
        if self.cls_token is None:
            if 'cls_token' in model_state_dict.keys():
                print('WARNING: Loading a model with a cls_token, but the current model does not have a cls_token. The cls_token will be ignored')
                del model_state_dict['cls_token']
        
        if self.bad_token is None:
            if 'bad_token' in model_state_dict.keys():
                print('WARNING: Loading a model with a bad_token, but the current model does not have a bad_token. The bad_token will be ignored')
                del model_state_dict['bad_token']

        model_dict = self.state_dict()
        model_dict.update(model_state_dict)
        self.load_state_dict(model_dict)
    
    
    def _normalize_embeddings_batch(self, embeddings_batch):
        input_pixel_mean = np.array([0.406, 0.456, 0.485])  # BGR
        input_pixel_std = np.array([0.225, 0.224, 0.299])  # BGR

        embeddings_batch = embeddings_batch.astype(np.float32) / 255.0
        embeddings_batch -= input_pixel_mean
        embeddings_batch /= input_pixel_std
        
        return embeddings_batch


    @staticmethod
    # @jit(nopython=True)
    def ltwh_to_ltrb(ltwh):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = torch.clone(ltwh)
        ret[..., 2:] += ret[..., :2]
        return ret


    def get_image_crops(self, image, bboxes, output_size=None, normalize=True):  # Like the one in embed_dataset.py (but for pytorch and multiple bboxes)
        if output_size is None:
            output_size = (self.expected_image_size[1], self.expected_image_size[0])

        crops = []
        for bbox in bboxes:  # Bboxes in format  x1y1x2y2
            # bbox = np.array([bbox[1], bbox[0], bbox[3], bbox[2]])
            crop = get_bbox_crop(im=image, bbox_real_scale=bbox, output_size=output_size, normalize=normalize, ghost_normalize=True)  # x1, y1, x2, y2
            crops.append(crop)
        
        if len(crops) == 0:
            crops = np.zeros([0, output_size[0], output_size[1], 3])  # We return an empty array
        else:
            crops = np.stack(crops, axis=0)

        return crops


class ReID_Encoder(nn.Module):
    # We will load the reid module Bag of Tricks (GHOST version)
    PRETRAINED_SIZE = (384, 128)

    def __init__(self, num_classes, device, pretrained_path=None, use_domain_adaptation=True, output_option='plain', trainable=False, use_checkpointing=True):
        super(ReID_Encoder, self).__init__()

        self.num_classes = num_classes  # num_classes doesn't really matter during inference, as we will be only interested in the features
        self.device = device
        self.pretrained_path = pretrained_path
        self.use_domain_adaptation = use_domain_adaptation
        self.output_option = output_option
        self.trainable = trainable
        self.use_checkpointing = use_checkpointing  # We have to set to false to avoid "UserWarning: None of the inputs have requires_grad=True. Gradients will be None"
        
        self.model = None  # We create the model in the build function
        self.embedding_size = None  # We populate this in the _get_encoder function
        self._get_encoder()

        # We want to use this as a feature extractor only, and we will not train it
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False
    

    def _get_encoder(self):
        model, sz_embed, optimizer_state_dict = load_net(nb_classes=self.num_classes, net_type="resnet50", neck=0, pretrained_path=self.pretrained_path, red=4, add_distractors=False, pool='max', use_checkpointing=self.use_checkpointing)

        self.model = model.to(self.device)
        self.embedding_size = sz_embed
    

    def get_features(self, frame):
        """
        Compute reid feature vectors
        """
        # We first assert that the frame size is what we expect
        assert frame.shape[2:] == ReID_Encoder.PRETRAINED_SIZE, "The frame size is not the expected one. Expected: {}, got: {}".format(ReID_Encoder.PRETRAINED_SIZE, frame.shape[2:])

        # Forward pass
        if self.trainable:  # If we want to train it, we do nothing in particular
            cls_probs, feats = self.model(frame, output_option=self.output_option)
        
        else:  # If we don't want to train it, we do stuff depending on the domain adaptation option
            if self.use_domain_adaptation:  # We want the batch_norms to be always in train mode
                if not self.model.training:
                    self.model.train()
            
            else:
                # If we don't use domain adaptation, we want the model to always be in eval mode (because we don't want to update the batch_norms), we are using it only as a feature extractor
                if self.model.training:
                    self.model.eval()

            with torch.no_grad():
                cls_probs, feats = self.model(frame, output_option=self.output_option)
            
            # We don't want gradients
            cls_probs = cls_probs.detach()
            feats = feats.detach()

        return cls_probs, feats
    

    def forward(self, x):
        out = self.get_features(x)
        return out
