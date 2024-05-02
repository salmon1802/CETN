# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import time

import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class CL4CTR(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="CL4CTR",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 FI_encode_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 alpha=1,
                 beta=0.01,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(CL4CTR, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.ep_dim = int(self.num_fields * (self.num_fields + 1) / 2) * self.embedding_dim
        self.ip_dim = int(self.num_fields * (self.num_fields + 1) / 2)
        self.flatten_dim = feature_map.sum_emb_out_dim()
        self.triu_node_index = nn.Parameter(torch.triu_indices(self.num_fields, self.num_fields, offset=0),
                                            requires_grad=False)
        self.mlp_prediction = MLP_Block(input_dim=self.flatten_dim,
                              output_dim=1,
                              hidden_units=hidden_units,
                              hidden_activations=hidden_activations,
                              output_activation=None,
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=1, dim_feedforward=128,
        #                                                 dropout=0.2)
        # self.FI_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.FI_encoder = MLP_Block(input_dim=self.flatten_dim,
                              output_dim=self.flatten_dim,
                              hidden_units=FI_encode_units,
                              hidden_activations=hidden_activations,
                              output_activation=None,
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)


        self.projector1 = nn.Linear(self.flatten_dim, embedding_dim)
        self.projector2 = nn.Linear(self.flatten_dim, embedding_dim)

        self.random_mask1 = nn.Dropout(0.1)
        self.random_mask2 = nn.Dropout(0.1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        feature_emb_flatten = feature_emb.flatten(start_dim=1)
        y_pred = self.mlp_prediction(feature_emb_flatten)

        feature_emb1 = self.random_mask1(feature_emb)
        feature_emb2 = self.random_mask2(feature_emb)


        # h1 = self.FI_encoder(feature_emb1).flatten(start_dim=1)
        # h2 = self.FI_encoder(feature_emb2).flatten(start_dim=1)

        h1 = self.FI_encoder(feature_emb1.flatten(start_dim=1))
        h2 = self.FI_encoder(feature_emb2.flatten(start_dim=1))

        h1 = self.projector1(h1)
        h2 = self.projector2(h2)

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "h1": h1, "h2": h2,
                       "feature_emb1": feature_emb1,
                       "feature_emb2": feature_emb2}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        h1 = return_dict["h1"]
        h2 = return_dict["h2"]
        feature_emb1 = return_dict["feature_emb1"]
        feature_emb2 = return_dict["feature_emb2"]
        cl_loss = self.cl_loss(h1, h2)
        feat_list_1 = feature_emb1.chunk(self.num_fields, dim=1)
        feat_list_2 = feature_emb2.chunk(self.num_fields, dim=1)
        alignment_loss = self.alignment_loss(feat_list_1, feat_list_2)
        uniformity_loss = self.uniformity_loss(feat_list_1, feat_list_2)
        loss = loss + self.alpha * cl_loss + self.beta * (alignment_loss + uniformity_loss)
        return loss

    def cl_loss(self, h1, h2):
        return torch.norm(h1.sub(h2), dim=1).pow_(2).mean()

    def alignment_loss(self, feat_list_1, feat_list_2):
        alignment_loss_sum = []
        for e1, e2 in zip(feat_list_1, feat_list_2):
            e1 = torch.squeeze(e1, dim=1)
            e2 = torch.squeeze(e2, dim=1)
            alignment_loss = torch.norm(e1.sub(e2), dim=1).pow_(2).mean()
            alignment_loss_sum.append(alignment_loss)
        alignment_loss_sum = torch.stack(alignment_loss_sum, dim=-1)
        return torch.mean(alignment_loss_sum)

    def uniformity_loss(self, feat_list_1, feat_list_2):
        cosine_similarity_loss = []
        for i, feat_1 in enumerate(feat_list_1):
            for j, feat_2 in enumerate(feat_list_2):
                if i != j:
                    feat_1 = torch.squeeze(feat_1, dim=1)
                    feat_2 = torch.squeeze(feat_2, dim=1)
                    cosine_sim = F.cosine_similarity(feat_1, feat_2).mean()
                    cosine_similarity_loss.append(cosine_sim)
        cosine_similarity_loss = torch.stack(cosine_similarity_loss, dim=-1)
        return torch.mean(cosine_similarity_loss)
