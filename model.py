import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)  # [64, 96, 207, 12]
        h = self.mlp(h)  # [64, 32, 207, 12]
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class Temporal_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if (k == 1):
            self.temporal_conv = nn.Identity()
        else:
            self.temporal_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2, dilation=1
            )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, x):
        out = self.pointwise_conv(self.temporal_conv(x))
        return out


class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):

        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Temporal_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Temporal_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Temporal_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # bs,dim,n
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)  # bs.n.dim

        out = out + out2
        return out
class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #temporal attention vector
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #spatial attention vector
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out

class stmffn(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, end_channels=512,
                 kernel_size=2, blocks=8):
        super(stmffn, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device


        self.filter1_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.wy1 = nn.Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=(1, 1))
        self.wy2 = nn.Conv2d(in_channels=64,
                            out_channels=32,
                            kernel_size=(1, 1))
        self.supports = supports
        self.psa = ParallelPolarizedSelfAttention(channel=32).to(device)




        receptive_field = 1
        self.device = device
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            #  print("================gcn_bool and add====================")
            if aptinit is None:
                # print("================aptinit===================")
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            # print("=================================model  if=================================================")
            # print(self.nodevec1)
            # print(self.nodevec2)
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
            # print("=================================model  else=================================================")
            # print(self.nodevec1)
        # print(self.nodevec2)
        # print("=================================model   stmffn=================================================")
        # print(self.nodevec1)
        # print(self.nodevec2)

        for i in range(blocks):
            # dilated convolutions
            additional_scope = kernel_size - 1
            new_dilation = 1
            self.filter1_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                out_channels=dilation_channels,
                                                kernel_size=(1, kernel_size), dilation=new_dilation))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            new_dilation *= 2
            receptive_field += additional_scope
            additional_scope *= 2
            if self.gcn_bool:
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1_skip = nn.Conv2d(in_channels=32,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2_skip = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def spatial_attention_cosAtt(self,x_input):
        '''
        Spatial attention cosAtt layer without dimention change.
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :param c_in: int, size of input channel.
        :param c_out: int, size of output channel.
        :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
        '''
        _batch_size_=64

        x = x_input
        _T_ = x.shape[3]
        _n_ = x.shape[2]
        x = torch.transpose(x, 2, 3)
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)

        x_result = torch.matmul(x, torch.transpose(x, 2,3 ))

        norm2_result = torch.matmul(norm2, torch.transpose(norm2, 2,3))


        self.dy_paras = nn.Parameter(torch.ones(_T_))
        self.softmax = nn.Softmax(-1)
        attention_beta = nn.Parameter(self.softmax(self.dy_paras)).cuda(0)
        cos = torch.divide(x_result, norm2_result + 1e-7)

        cos_ = torch.multiply(attention_beta, cos)
        P = torch.sigmoid(cos_)

        output = torch.matmul(P, x)
        output = torch.transpose(output, 2, 3)

        return output


    def forward(self, input):  # input来自engine中trainer的输入
        # input:[batch_siza, feature_size, node number, in_len]
        # output:[batch_siza, feature_size, node number, out_dim]
        # print('====stmffn====')

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)

        # calculate the current adaptive adj m
        # atrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # print("=================================model  forward if=================================================")
            # print(self.nodevec1)
            # print(self.nodevec2)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]


        # # feature extraction block
        for i in range(self.blocks):
            x1 = x

            # gated graph convolution module
            xt1 = self.gconv[i](x1, new_supports)
            xt1 = torch.tanh(xt1)
            xt2 = self.gconv[i](x1, new_supports)
            xt2 = torch.sigmoid(xt2)
            residual1 = xt1 * xt2
            cosAttresidual1=self.spatial_attention_cosAtt(x1)
            residual1 = cosAttresidual1 * torch.sigmoid(residual1)



            filter1 = self.filter1_convs[i](x1)
            # multi-scale attention module
            self.sa = MUSEAttention(d_model=filter1.shape[3], d_k=filter1.shape[3], d_v=filter1.shape[3], h=8).to(self.device)
            residual20 = filter1.sum(dim=1)
            output = self.sa(residual20, residual20, residual20)
            residual2 = torch.einsum('btnf,bnm->btnm', [filter1, output])
            residual2 = self.wy1(residual2)
            residual2 = self.wy2(residual2)




            x = residual1[:, :, -residual2.size(2):, -residual2.size(3):] + residual2
            x = self.psa(x)
            x = self.bn[i](x)


        # prediction layer
        x = F.relu(x)
        x = F.relu(self.end_conv_1_skip(x))
        x = self.end_conv_2_skip(x)
        return x





