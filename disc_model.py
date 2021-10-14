from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import numpy as np
# from tensorboardX import SummaryWriter
# writer1 = SummaryWriter('run/exp')
from torch.nn.init import xavier_uniform_

import create_svg



device = torch.device('cuda:3')
max_len_src = 16
max_len_tgt = 16
d_model = 512

def quantize(input, n_range = 256):
    intv = 500. / n_range
    code = input / intv
    code = (code+0.5).astype(np.long)
    # one_hot_code = np.zeros(n_range, dtype=np.float32)
    # one_hot_code[one_hot] = 1
    return code

def rev_quantize(prob, n_range = 256):
    intv = 500. / n_range
    output = np.argmax(prob) * intv
    return output

class PositionalEncodingAbsolute(nn.Module):
    def __init__(self, max_len):
        super(PositionalEncodingAbsolute, self).__init__()
        self.pos_encoding = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        for i in pos:
            self.pos_encoding[i, 0::2] = torch.sin(i / (10000 ** (_2i / d_model)))
            self.pos_encoding[i, 1::2] = torch.cos(i / (10000 ** (_2i / d_model)))
        self.pos_encoding.requires_grad = False

    def forward(self, x):
        return x + self.pos_encoding

class PositionalEncodingLUT(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)

class SVGEmbedding(nn.Module):
    def __init__(self):
        super(SVGEmbedding, self).__init__()
        self.command_emb = nn.Embedding(4, d_model)
        self.coord_emb = nn.Embedding(256, 64)
        # self.color_emb = nn.Embedding(256, )
        self.emb_fcn = nn.Linear(64 * 7, d_model)
        # self.pos_encoding = PositionalEncodingLUT(d_model, max_len=16)
        self.pos_encoding = PositionalEncodingAbsolute(16)

    def forward(self, commands, x):
        bs, code, _ = x.shape
        cmd_emb = self.command_emb(commands)
        coor_emb = self.coord_emb(x)
        fcn_emb = self.emb_fcn(coor_emb.view(bs, code, -1))
        # print(cmd_emb.shape)
        # print(fcn_emb.shape)
        src =  cmd_emb + fcn_emb
        # src = self.pos_encoding(src)
        #position embedding
        return src

class Probabilityffn(nn.Module):
    def __init__(self):
        super(Probabilityffn, self).__init__()
        self.command_ffn = nn.Linear(d_model, 4)
        self.coord_ffn = nn.Linear(d_model, 7 * 256)

    def forward(self, out):
        bs, svg_len, d_svg = out.shape
        p_command = self.command_ffn(out)
        p_coord = self.coord_ffn(out)
        p_coord = p_coord.reshape(bs, svg_len, 7, 256)
        return p_command, p_coord

class SVGData(Dataset):
    def __init__(self, n_svgs = 1000, refphase = False):
        self.svgs, self.texsvgs = create_svg.TrainData_SVG(n_svgs)
        self.len_svg = len(self.svgs)
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt
        self.n_embed_primitives = 7
        self.length = len(self.svgs)
        self.eval = refphase
# start 1
# end 2
# used 3
# unused 0
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        src = np.zeros((self.max_len_src, self.n_embed_primitives), dtype=np.long)
        srcode = self.svgs[index].svgparams()
        srcode = quantize(srcode)
        src[1:len(srcode)+1] = srcode

        tgt = np.zeros((self.max_len_tgt, self.n_embed_primitives), dtype=np.long)
        tgcode = self.texsvgs[index].svgparams()
        tgcode = quantize(tgcode)
        tgt[1:len(tgcode)+1] = tgcode

        gt = np.zeros((self.max_len_tgt, self.n_embed_primitives), dtype=np.long)
        gt[0:len(tgcode)] = tgcode
        

        commandsrc = np.zeros(self.max_len_src, dtype=np.long)
        commandsrc[0] = 1
        commandsrc[1:len(srcode)+1] = 3
        commandsrc[len(srcode)+1] = 2
        commandtgt = np.zeros(self.max_len_tgt, dtype=np.long)
        commandtgt[0] = 1
        commandtgt[1:len(tgcode)+1] = 3
        commandtgt[len(tgcode)+1] = 2
        commandgt = np.zeros(self.max_len_tgt, dtype=np.long)
        commandgt[0:len(tgcode)] = 3
        commandgt[len(tgcode)] = 2


        srcpad = np.zeros(self.max_len_src, dtype=np.bool)
        srcpad[len(srcode)+2:] = True
        tgtpad = np.zeros(self.max_len_tgt, dtype=np.bool)
        tgtpad[len(tgcode)+2:] = True
        return commandsrc, commandtgt, commandgt, src, tgt, gt, srcpad, tgtpad


def SVGLoader(batch_size = 4, num_workers = 8, n_svgs = 1000):
    data = SVGData(n_svgs)
    return DataLoader(data, batch_size = batch_size, num_workers = num_workers, shuffle = True)

class svg_transformer(nn.Module):
    def __init__(self, d_model):
        super(svg_transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=1, num_encoder_layers=1, 
            num_decoder_layers=1, dim_feedforward=512, dropout=0.1, 
            activation='relu', custom_encoder=None, custom_decoder=None, batch_first=True)
        self.transformer._reset_parameters()
        self.embed_input = SVGEmbedding()
        self.embed_output = SVGEmbedding()
        self.out_ffn = Probabilityffn()
        
        

    def forward(self, cmdsrc, cmdtgt, src, tgt, src_mask = None, tgt_mask = None, srcpad = None, tgtpad = None):
        src = self.embed_input(cmdsrc, src)
        tgt = self.embed_output(cmdtgt, tgt)
        # src = src.transpose(0, 1)
        # tgt = tgt.transpose(0, 1)
        out = self.transformer.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask=srcpad, tgt_key_padding_mask=tgtpad)
        # out = out.transpose(0, 1).contiguous()
        # bs = out.shape[0]
        # out = out.view(bs, 512*16)
        # out = self.ff(out)
        # out = out.view(bs, 16, 7)
        log_cmd, log_coo  = self.out_ffn(out)
        # log_cmd, log_coo = torch.sigmoid(log_cmd), torch.sigmoid(log_coo)
        return log_cmd, log_coo

    def forward_encoder(self, cmdsrc, src, src_mask = None, srcpad = None):
        src = self.embed_input(cmdsrc, src)
        memory = self.transformer.encoder(src, mask = src_mask, src_key_padding_mask = srcpad)
        return memory
    
    def forward_decoder(self, cmdtgt, tgt, memory, tgt_mask = None, tgtpad = None):
        tgt = self.embed_output(cmdtgt, tgt)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgtpad)
        log_cmd, log_coo  = self.out_ffn(output)
        return log_cmd, log_coo
        
def train(net, data):
    pass

def validate(net, data):
    pass

def translate_svg(net : svg_transformer, data):
    commandsrc, commandtgt, commandgt, src, tgt, gt, srcpad, tgtpad = data
    commandsrc = commandsrc.to(device)
    commandtgt = commandtgt.to(device)
    commandgt = commandgt.to(device)
    src = src.to(device)
    tgt = tgt.to(device)
    gt = gt.to(device)
    srcpad = srcpad.to(device)
    tgtpad = tgtpad.to(device)
    cmd_output = torch.zeros_like(commandtgt).to(device)
    tgt_output = torch.zeros_like(tgt).to(device)
    cmd_output[0][0] = commandtgt[0][0]
    tgt_output[0][0] = tgt[0][0]
    memory = net.forward_encoder(commandsrc, src, srcpad = srcpad)
    with torch.no_grad():
        for i in range(max_len_tgt-1):
            output_cmd, output_coo = net.forward_decoder(cmd_output[:, 0:(i+1)], tgt_output[:, 0:(i+1)], memory)
            import pdb; pdb.set_trace()
            # print(cmd_output.shape)
            # print(output_cmd.shape)
            # print(output_coo.shape)
            # print(torch.argmax(output_cmd[0][i], dim=0))
            # print(cmd_output.shape)
            cmd_output[0][i+1] = torch.argmax(output_cmd[0][i], dim=0)
            tgt_output[0][i+1] = torch.argmax(output_coo[0][i], dim=1)
    
    print(gt)
    print(commandtgt)
    print(cmd_output)
    print(tgt_output)
    return tgt_output
def w_init(net):
    # print(net.named_modules())
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # print(m)
            xavier_uniform_(m.weight)

if __name__ == '__main__':
    net = svg_transformer(512)
    net.to(device)
    w_init(net)
    # net.load_state_dict(torch.load('svt.pt'))
    TrainData = SVGLoader(batch_size=100, n_svgs=1000)
    TestData = SVGLoader(batch_size=10, n_svgs=100)
    TranslateData = SVGLoader(batch_size=1, n_svgs=1)
    # visualize(TestData, net)
    epoch = 500
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    print('---Apply Adam optimizer---')
    # criterion = nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    print('--Loss: CrossEntropyLoss--')
    print('--train start--')
    for ep in range(epoch):
        
        # break
        loss_ep = 0
        net.train()
        for i, data in enumerate(TrainData):
            # import pdb; pdb.set_trace();
            commandsrc, commandtgt, commandgt, src, tgt, gt, srcpad, tgtpad = data
            commandsrc = commandsrc.to(device)
            commandtgt = commandtgt.to(device)
            commandgt = commandgt.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            gt = gt.to(device)
            srcpad = srcpad.to(device)
            tgtpad = tgtpad.to(device)
            mask_src = None
            mask_tgt = net.transformer.generate_square_subsequent_mask(16).to(device)
            # mask_tgt = torch.triu(torch.ones((16, 16), dtype=torch.bool), diagonal=1).to(device)
            log_cmd, log_coo = net.forward(commandsrc, commandtgt, src, tgt, mask_src, mask_tgt, srcpad, tgtpad)
            loss_cmd = criterion(log_cmd.view(-1, 4), commandgt.view(-1))
            loss_coo = criterion(log_coo.view(-1, 256), tgt.view(-1))
            loss = loss_cmd + loss_coo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            # if (ep == 10):
                # import pdb; pdb.set_trace()
            # print(loss.item())
        # writer1.add_scalar('loss', loss_ep, global_step=ep)
        print(f'train loss ep{ep}:{loss_ep}')
        # print(out)
        # print(gt)
        if (ep == 50):
            # break
            # net.eval()
            for i, data in enumerate(TrainData):
                translate_svg(net, data)
                exit()
        continue
        # net.eval()
        with torch.no_grad():
            if (ep % 10 != 0):
                continue
            loss_ep = 0
            for i, data in enumerate(TestData):
                commandsrc, commandtgt, commandgt, src, tgt, gt, srcpad, tgtpad = data
                commandsrc = commandsrc.to(device)
                commandtgt = commandtgt.to(device)
                commandgt = commandgt.to(device)
                src = src.to(device)
                tgt = tgt.to(device)
                gt = gt.to(device)
                srcpad = srcpad.to(device)
                tgtpad = tgtpad.to(device)
                mask_src = None
                mask_tgt = net.transformer.generate_square_subsequent_mask(16).to(device)
                # mask_tgt = torch.triu(torch.ones((16, 16), dtype=torch.bool), diagonal=1).to(device)
                log_cmd, log_coo = net.forward(commandsrc, commandtgt, src, tgt, mask_src, mask_tgt, srcpad, tgtpad)
                loss_cmd = criterion(log_cmd.view(-1, 4), commandtgt.view(-1))
                loss_coo = criterion(log_coo.view(-1, 256), tgt.view(-1))
                loss = loss_cmd + loss_coo
                loss_ep += loss.item()
            print(f'test loss ep{ep}:{loss_ep}')
    
    # visualize(TrainData, net)
    # print('--save model--')
    # torch.save(net.state_dict(), 'svt.pt')