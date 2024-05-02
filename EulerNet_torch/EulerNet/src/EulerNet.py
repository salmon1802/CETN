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


from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression


class EulerNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="EulerNet",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 order_list=[10, 10],
                 dropout_explicit=0,
                 dropout_implicit=0,
                 batch_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(EulerNet, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        num_fields = feature_map.num_fields
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        shape_list = [embedding_dim * num_fields] + \
                          [num_neurons * embedding_dim for num_neurons in order_list]
        interaction_shapes = []
        for inshape, outshape in zip(shape_list[:-1], shape_list[1:]):
            interaction_shapes.append(EulerInteractionLayer(dropout_explicit=dropout_explicit,
                                                            dropout_implicit=dropout_implicit,
                                                            inshape=inshape,
                                                            outshape=outshape,
                                                            embedding_dim=embedding_dim,
                                                            batch_norm=batch_norm))
        self.Euler_interaction_layers = nn.Sequential(*interaction_shapes)
        self.mu = nn.Parameter(torch.ones(1, num_fields, 1))
        self.fc = nn.Linear(shape_list[-1], 1)
        nn.init.xavier_normal_(self.fc.weight)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        r, p = self.mu * torch.cos(feature_emb), self.mu * torch.sin(feature_emb)
        o_r, o_p = self.Euler_interaction_layers((r, p))
        o_r, o_p = o_r.reshape(o_r.shape[0], -1), o_p.reshape(o_p.shape[0], -1)
        re, im = self.fc(o_r), self.fc(o_p)
        y_pred = re + im
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class EulerInteractionLayer(nn.Module):
    def __init__(self,
                 dropout_explicit,
                 dropout_implicit,
                 inshape,
                 outshape,
                 embedding_dim,
                 batch_norm):
        super().__init__()
        self.inshape, self.outshape = inshape, outshape
        self.feature_dim = embedding_dim
        self.apply_norm = batch_norm

        # Initial assignment of the order vectors, which significantly affects the training effectiveness of the model.
        # We empirically provide two effective initialization methods here.
        # How to better initialize is still a topic to be further explored.
        if inshape == outshape:
            init_orders = torch.eye(inshape // self.feature_dim, outshape // self.feature_dim)
        else:
            init_orders = torch.softmax(torch.randn(inshape // self.feature_dim, outshape // self.feature_dim) / torch.tensor(0.01),
                                        dim=0)

        self.inter_orders = nn.Parameter(init_orders)
        self.im = nn.Linear(inshape, outshape)
        nn.init.normal_(self.im.weight, mean=0, std=0.01)

        self.bias_lam = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * torch.tensor(0.01))
        self.bias_theta = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * torch.tensor(0.01))

        self.drop_ex = nn.Dropout(p=dropout_explicit)
        self.drop_im = nn.Dropout(p=dropout_implicit)
        self.norm_r = nn.LayerNorm([self.feature_dim])
        self.norm_p = nn.LayerNorm([self.feature_dim])

    def forward(self, complex_features):
        r, p = complex_features

        lam = r ** 2 + p ** 2 + 1e-8
        theta = torch.atan2(p, r)
        lam, theta = lam.reshape(lam.shape[0], -1, self.feature_dim), theta.reshape(theta.shape[0], -1,
                                                                                    self.feature_dim)
        lam = torch.tensor(0.5) * torch.log(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta = lam @ (self.inter_orders) + self.bias_lam, theta @ (self.inter_orders) + self.bias_theta
        lam = torch.exp(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)

        r, p = r.reshape(r.shape[0], -1), p.reshape(p.shape[0], -1)
        r, p = self.drop_im(r), self.drop_im(p)
        r, p = self.im(r), self.im(p)
        r, p = torch.relu(r), torch.relu(p)
        r, p = r.reshape(r.shape[0], -1, self.feature_dim), p.reshape(p.shape[0], -1, self.feature_dim)

        o_r, o_p = r + lam * torch.cos(theta), p + lam * torch.sin(theta)
        o_r, o_p = o_r.reshape(o_r.shape[0], -1, self.feature_dim), o_p.reshape(o_p.shape[0], -1, self.feature_dim)
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)
        return o_r, o_p