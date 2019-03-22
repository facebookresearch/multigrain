# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from multigrain.backbones import BackBone
import torch.utils.model_zoo as model_zoo
from multigrain.modules.layers import Layer, Select, IntegratedLinear
from multigrain.modules import DistanceWeightedSampling


__all__ = ['multigrain']


model_urls = {
    ('multigrain_resnet50'): '',
}


class MultiGrain(BackBone):
    """
    Implement MultiGrain by changing the pooling layer of the backbone into GeM pooling with exponent p,
    and adding DistanceWeightedSampling for the margin loss.
    """
    def __init__(self, backbone, p=3.0, include_sampling=True, learn_p=False, **kwargs):
        super().__init__(backbone, **kwargs)
        if not torch.is_tensor(p):
            p = torch.tensor(p)
            if learn_p:
                p.requires_grad = True
        add_bias = self.classifier.bias is not None
        self.pool = Layer('gem', p=p, add_bias=add_bias)
        if add_bias:
            if self.pre_classifier is not None:
                self.pre_classifier = Select(self.pre_classifier, -1)
            self.classifier = IntegratedLinear(self.classifier)
        self.normalize = Layer('l2n')
        if include_sampling:
            self.weighted_sampling = DistanceWeightedSampling()

    def forward(self, input, instance_target=None, **kwords):
        if isinstance(instance_target, list):
            instance_target = torch.stack(instance_target)
        output_dict = {'instance_target': instance_target}
        output_dict['embedding'], output_dict['classifier_output'] = super().forward(input, **kwords)
        output_dict['normalized_embedding'] = self.normalize(output_dict['embedding'])

        if hasattr(self, 'weighted_sampling') and instance_target is not None:
            sampled = self.weighted_sampling(output_dict['normalized_embedding'], instance_target)
            output_dict.update(sampled)

        return output_dict


def get_multigrain(backbone='resnet50', pretrained=None, pretrained_backbone=None, **kwargs):
    kwargs['pretrained'] = pretrained_backbone
    model = MultiGrain(backbone, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multigrain_' + backbone]))
    return model
