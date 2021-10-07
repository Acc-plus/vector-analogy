from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import numpy as np
# from tensorboardX import SummaryWriter
# writer1 = SummaryWriter('run/exp')


import create_svg

def quantize(input, n_range):
    intv = 1. / n_range
    one_hot = int(input / intv)
    one_hot_code = np.zeros(n_range, dtype=np.float32)
    one_hot_code[one_hot] = 1
    return one_hot_code

class SVGData(Dataset):
    def __init__(self, n_svgs = 1000):
        self.svgs, self.texsvgs = create_svg.TrainData_SVG(n_svgs)
        self.len_svg = len(self.svgs)
        self.max_len_src = 16
        self.max_len_tgt = 16
        self.n_embed_primitives = 7
        self.length = len(self.svgs)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        src = np.zeros((self.max_len_src, self.n_embed_primitives), dtype=np.float32)
        srcode = self.svgs[index].svgparams()
        src[0:len(srcode)] = srcode
        tgt = np.zeros((self.max_len_tgt, self.n_embed_primitives), dtype=np.float32)
        texcode = self.texsvgs[index].svgparams()
        tgt[0:len(texcode)] = texcode
        gt = np.zeros((self.max_len_tgt, self.n_embed_primitives), dtype=np.float32)
        gt[0:len(texcode)] = texcode
        gtmask = np.zeros((self.max_len_tgt, self.n_embed_primitives), dtype=np.float32)
        gtmask[0:len(texcode)] = 1
        return src/np.float32(500), tgt/np.float32(500), gt/np.float32(500), gtmask

def SVGLoader(batch_size = 4, num_workers = 8, n_svgs = 1000):
    data = SVGData(n_svgs)
    return DataLoader(data, batch_size = batch_size, num_workers = num_workers, shuffle = True)

class svg_transformer(nn.Module):
    def __init__(self, d_model):
        super(svg_transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=8, num_encoder_layers=6, 
            num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
            activation='relu', custom_encoder=None, custom_decoder=None)

        self.ff = nn.Linear(d_model*16, 7*16)
        self.w_x = nn.Parameter(torch.randn(d_model, 1))
        self.w_coord = nn.Parameter(torch.randn(d_model, 7*d_model))
        self.register_parameter('wx', self.w_x)
        self.register_parameter('wc', self.w_coord)
        self.pos_encoding = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        for i in pos:
            self.pos_encoding[i, 0::2] = torch.sin(i / (10000 ** (_2i / d_model)))
            self.pos_encoding[i, 1::2] = torch.cos(i / (10000 ** (_2i / d_model)))
        self.pos_encoding.requires_grad = False
    def embed_params(self, src):
        bs = src.shape[0]
        nparam = src.shape[1]
        src = src.unsqueeze(2)
        src = torch.einsum('do,bkoj->bkdj', self.w_x, src)
        src = src.reshape(bs, nparam, 7*d_model)
        src = torch.einsum('dq,bkq->bkd', self.w_coord, src)
        return src

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        # print(self.w_rec_embedding.shape)
        # print(src.shape)
        # src = self.w_rec_embedding @ src
        src = self.embed_params(src)
        tgt = self.embed_params(tgt)
        src += self.pos_encoding
        tgt += self.pos_encoding
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # print(src.shape)
        # print(tgt.shape)
        # print(mask_tgt.shape)
        out = self.transformer.forward(src, tgt, src_mask, tgt_mask)
        
        out = out.transpose(0, 1).contiguous()
        bs = out.shape[0]
        # print(out.shape)
        out = out.view(bs, 512*16)
        out = self.ff(out)
        out = out.view(bs, 16, 7)
        
        return out


n_enc = 0
def visualize(TestData, net):
    global n_enc
    device = torch.device('cuda')
    net.eval()
    print('--visualize--')
    with torch.no_grad():
        loss_ep = 0
        for i, data in enumerate(TestData):
            src, tgt, gt, gtmask = data
            src, tgt, gt, gtmask = src.to(device), tgt.to(device), gt.to(device), gtmask.to(device)
            mask_src = None
            mask_tgt = torch.triu(torch.ones((16, 16), dtype=torch.bool), diagonal=1).to(device)
            out = net.forward(src, tgt, mask_src, mask_tgt)
            # loss = criterion(out, gt)
            bs = out.shape[0]
            src = src.cpu().numpy().astype(np.float)*500
            out = out.cpu().numpy().astype(np.float)*500
            for j in range(bs):
                svgi = create_svg.SVG()
                svgo = create_svg.SVG()
                for k in range(5):
                    sr = src[j][k]
                    ou = out[j][k]
                    svgi.primitives.append(create_svg.rectangle(sr[0], sr[1], sr[2], sr[3], [sr[4], sr[5], sr[6]]))
                    svgo.primitives.append(create_svg.rectangle(ou[0], ou[1], ou[2], ou[3], [ou[4], ou[5], ou[6]]))
                create_svg.draw_svg(svgi, f'output/{n_enc}i.svg')
                create_svg.draw_svg(svgo, f'output/{n_enc}o.svg')
                n_enc += 1
    



if __name__ == '__main__':
    max_len = 16
    d_model = 512
    device = torch.device('cuda')
    net = svg_transformer(512)
    net.to(device)
    net.load_state_dict(torch.load('svt.pt'))
    TrainData = SVGLoader(batch_size=100, n_svgs=1000)
    TestData = SVGLoader(batch_size=100, n_svgs=10)
    # visualize(TestData, net)
    epoch = 150
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.MSELoss()
    for ep in range(epoch):
        loss_ep = 0
        print('--train--start--')
        net.train()
        for i, data in enumerate(TrainData):
            
            src, tgt, gt, gtmask = data
            src, tgt, gt, gtmask = src.to(device), tgt.to(device), gt.to(device), gtmask.to(device)
            mask_src = None
            mask_tgt = torch.triu(torch.ones((16, 16), dtype=torch.bool), diagonal=1).to(device)
            # pos_encoding = torch.zeros(max_len, d_model, device=device)
            # pos = torch.arange(0, max_len, device=device)
            # _2i = torch.arange(0, d_model, step=2, device=device).float()
            # for i in pos:
            #     pos_encoding[i, 0::2] = torch.sin(i / (10000 ** (_2i / d_model)))
            #     pos_encoding[i, 1::2] = torch.cos(i / (10000 ** (_2i / d_model)))
            # src += pos_encoding
            # tgt += pos_encoding
            # print(src)
            # print(gt)
            # print(src.shape)
            # print(tgt.shape)
            out = net.forward(src, tgt, mask_src, mask_tgt)
            # print(out.shape)
            # out = out.transpose(0,1)
            # print(gt.shape)
            loss = (gtmask * (out - gt)**2).mean()
            # loss = criterion(out, gt)

            # print(out)
            # exit(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            # print(loss.item())
        # writer1.add_scalar('loss', loss_ep, global_step=ep)
        print(f'train loss ep{ep}:{loss_ep}')
        # print(out)
        # print(gt)
        net.eval()
        with torch.no_grad():
            if (ep % 10 != 0):
                continue
            loss_ep = 0
            for i, data in enumerate(TestData):
                src, tgt, gt, gtmask = data
                src, tgt, gt, gtmask = src.to(device), tgt.to(device), gt.to(device), gtmask.to(device)
                mask_src = None
                mask_tgt = torch.triu(torch.ones((16, 16), dtype=torch.bool), diagonal=1).to(device)
                out = net.forward(src, tgt, mask_src, mask_tgt)
                # loss = criterion(out, gt)
                loss = (gtmask * (out - gt)**2).mean()
                loss_ep += loss.item()
            print(f'test loss ep{ep}:{loss_ep}')

    torch.save(net.state_dict(), 'svt.pt')
    
    # net.load_state_dict(torch.load('svt.pt'))