import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LSCLinear, SplitLSCLinear

class IncreTouchVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreTouchVisualNet, self).__init__()
        self.args = args
        self.num_classes = step_out_class_num
        self.touch_proj = nn.Linear(768, 768)
        self.visual_proj = nn.Linear(768, 768)
        self.attn_touch_proj = nn.Linear(768, 768)
        self.attn_visual_proj = nn.Linear(768, 768)

        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)

    def forward(self, visual=None, touch=None, out_logits=True, out_features=False, out_features_norm=False,
                out_feature_before_fusion=False, out_attn_score=False, AFC_train_out=False):

        visual = visual.view(visual.shape[0], 8, -1, 768)
        touch = touch.view(touch.shape[0], 8, -1, 768)

        spatial_attn_score, temporal_attn_score = self.touch_visual_attention(touch, visual)

        visual_pooled_feature = torch.sum(spatial_attn_score * visual, dim=2)
        visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)

        touch_pooled_feature = torch.sum(spatial_attn_score * touch, dim=2)
        touch_pooled_feature = torch.sum(temporal_attn_score * touch_pooled_feature, dim=1)

        touch_feature = F.relu(self.touch_proj(touch_pooled_feature))
        visual_feature = F.relu(self.visual_proj(visual_pooled_feature))
        touch_visual_features = touch_feature + visual_feature

        logits = self.classifier(touch_visual_features)
        outputs = ()
        if AFC_train_out:
            touch_feature.retain_grad()
            visual_feature.retain_grad()
            visual_pooled_feature.retain_grad()
            touch_pooled_feature.retain_grad()
            outputs += (logits, touch_pooled_feature, visual_pooled_feature, touch_feature, visual_feature)
            return outputs
        else:
            if out_logits:
                outputs += (logits,)
            if out_features:
                if out_features_norm:
                    outputs += (F.normalize(touch_visual_features),)
                else:
                    outputs += (touch_visual_features,)
            if out_feature_before_fusion:
                outputs += (F.normalize(touch_feature), F.normalize(visual_feature))
            if out_attn_score:
                outputs += (spatial_attn_score, temporal_attn_score)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs

    def touch_visual_attention(self, touch_features, visual_features):

        proj_touch_features = torch.tanh(self.attn_touch_proj(touch_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,ijkd->ijkd", [proj_visual_features, proj_touch_features])
        # (BS, 8, 14*14, 768)
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)
        spatial_attned_proj_touch_features = torch.sum(spatial_attn_score * proj_touch_features, dim=2)
        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,ijd->ijd",[spatial_attned_proj_visual_features, spatial_attned_proj_touch_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score

    def incremental_classifier(self, numclass):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, numclass, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias