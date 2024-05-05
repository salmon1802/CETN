# =========================================================================
# Copyright (C) 2024 salmon1802li@gmail.com
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
import sys

sys.path.append("/mnt/public/lhh/code/")
import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


# Contrast-enhanced MLP Model for CTR
# 不加显式噪声而是不同的激活函数
class CETN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="CETN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 perturbed=False,
                 eps=0,
                 alpha=0.2,
                 beta=0.2,
                 delta=0.2,
                 cl_temperature=0.2,
                 through=True,
                 fi_hidden_units=[64, 64, 64],
                 w_hidden_units=[64, 64, 64],
                 hidden_activations=['relu'],
                 W_net_dropout=0,
                 V_net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(CETN, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.through = through
        self.cl_temperature = cl_temperature
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.ep_dim = int(self.num_fields * (self.num_fields + 1) / 2) * self.embedding_dim
        self.ip_dim = int(self.num_fields * (self.num_fields + 1) / 2)
        self.perturbed = perturbed
        self.eps = eps
        self.flatten_dim = feature_map.sum_emb_out_dim()
        self.triu_node_index = nn.Parameter(torch.triu_indices(self.num_fields, self.num_fields, offset=0),
                                            requires_grad=False)
        self.mlp1 = MLP_Block(input_dim=self.ep_dim,
                              output_dim=None,
                              hidden_units=fi_hidden_units,
                              hidden_activations=hidden_activations[0],
                              output_activation=None,
                              dropout_rates=V_net_dropout,
                              batch_norm=batch_norm)
        self.W1 = MLP_Block(input_dim=self.ep_dim,
                            output_dim=1,
                            hidden_units=w_hidden_units,
                            hidden_activations='leaky_relu',
                            output_activation='leaky_relu',
                            dropout_rates=W_net_dropout,
                            batch_norm=batch_norm)
        self.mlp2 = MLP_Block(input_dim=self.ip_dim,
                              output_dim=None,
                              hidden_units=fi_hidden_units,
                              hidden_activations=hidden_activations[1],
                              output_activation=None,
                              dropout_rates=V_net_dropout,
                              batch_norm=batch_norm)
        self.W2 = MLP_Block(input_dim=self.ip_dim,
                            output_dim=1,
                            hidden_units=w_hidden_units,
                            hidden_activations='leaky_relu',
                            output_activation='leaky_relu',
                            dropout_rates=W_net_dropout,
                            batch_norm=batch_norm)
        self.mlp3 = MLP_Block(input_dim=self.flatten_dim,
                              output_dim=None,
                              hidden_units=fi_hidden_units,
                              hidden_activations=hidden_activations[2],
                              output_activation=None,
                              dropout_rates=V_net_dropout,
                              batch_norm=batch_norm)
        self.W3 = MLP_Block(input_dim=self.flatten_dim,
                            output_dim=1,
                            hidden_units=w_hidden_units,
                            hidden_activations='leaky_relu',
                            output_activation='leaky_relu',
                            dropout_rates=W_net_dropout,
                            batch_norm=batch_norm)
        self.p1 = nn.Linear(fi_hidden_units[-1], 1, bias=True)
        self.p2 = nn.Linear(fi_hidden_units[-1], 1, bias=True)
        self.p3 = nn.Linear(fi_hidden_units[-1], 1, bias=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        embs_ep, embs_ip, embs_flatten = self.ep_ip_flatten_emb(feature_emb)
        if self.perturbed:
            random_noise = torch.rand_like(embs_ep).to(feature_emb.device)
            embs_ep += torch.sign(embs_ep) * F.normalize(random_noise, dim=-1) * self.eps
            random_noise = torch.rand_like(embs_ip).to(feature_emb.device)
            embs_ip += torch.sign(embs_ip) * F.normalize(random_noise, dim=-1) * self.eps
        W3 = self.W3(embs_flatten)
        X3 = self.mlp3(embs_flatten)
        W1 = self.W1(embs_ep)
        X1 = self.mlp1(embs_ep)
        W2 = self.W2(embs_ip)
        X2 = self.mlp2(embs_ip)
        if self.through:
            X1 = X1 + X3
            X2 = X2 + X3
        y_pred = W1*self.p1(X1) + W2*self.p2(X2) + W3*self.p3(X3)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "X1": X1, "X2": X2, "X3": X3}
        return return_dict

    def ep_ip_flatten_emb(self, h):
        emb1 = torch.index_select(h, 1, self.triu_node_index[0])
        emb2 = torch.index_select(h, 1, self.triu_node_index[1])
        embs_ep = emb1 * emb2
        embs_ip = torch.sum(embs_ep, dim=-1)
        embs_ep = embs_ep.view(-1, self.ep_dim)
        embs_flatten = h.flatten(start_dim=1)
        return embs_ep, embs_ip, embs_flatten

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        X1 = return_dict["X1"]
        X2 = return_dict["X2"]
        X3 = return_dict["X3"]
        residual_loss_13 = self.CosineSimilarityLoss(X1, X3)
        residual_loss_23 = self.CosineSimilarityLoss(X2, X3)
        cl_loss = self.InfoNCE(X1, X2, cl_temperature=self.cl_temperature)
        loss = loss + self.alpha * cl_loss + self.beta * residual_loss_13 + self.delta * residual_loss_23
        return loss

    def InfoNCE(self, embedding_1, embedding_2, cl_temperature):
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)

        pos_score = torch.exp(torch.tensor(1.0) / cl_temperature)

        ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / cl_temperature).sum(dim=1)

        loss = - torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(loss)

    def CosineSimilarityLoss(self, embedding_1, embedding_2):
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)

        cosine_sim = F.cosine_similarity(embedding_1, embedding_2)
        cosine_similarity_loss = 1.0 - cosine_sim
        return torch.mean(cosine_similarity_loss)
