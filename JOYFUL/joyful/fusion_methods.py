import torch
from torch import nn
from torch.nn import functional as F


class AutoFusion(nn.Module):
    def __init__(self, input_features):
        super(AutoFusion, self).__init__()
        self.input_features = input_features #passed as 1380 in JOYFUL
        # now passing embedding dims as the input feature length. 

        self.fuse_inGlobal = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fuse_outGlobal = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_features)
        )

        self.fuse_inInter = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fuse_outInter = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_features)
        )

        self.criterion = nn.MSELoss()

        self.projectA = nn.Linear(100, 460)
        self.projectT = nn.Linear(768, 460)
        self.projectV = nn.Linear(512, 460)
        self.projectB = nn.Sequential(
            nn.Linear(460, 460),
        )

    def forward(self, a=None, t=None, v=None):
        B = self.projectB(torch.ones(460))
        if a is not None:
            A = self.projectA(a)
            BA = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), A), dim=1)
            bba = torch.mm(BA, torch.unsqueeze(A, dim=1)).squeeze(1)
        else: 
            bba= None
        if t is not None:
            T = self.projectT(t)
            BT = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), T), dim=1)
            bbt = torch.mm(BT, torch.unsqueeze(T, dim=1)).squeeze(1)
        else: 
            bbt= None
        if v is not None:
            V = self.projectV(v)
            BV = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), V), dim=1)
            bbv = torch.mm(BV, torch.unsqueeze(V, dim=1)).squeeze(1)
        else: 
            bbv= None
        print(a.shape, t.shape)
        globalInput= torch.cat([tensor for tensor in (a,t,v) if tensor is not None])
        globalCompressed = self.fuse_inGlobal(globalInput) 
        globalLoss = self.criterion(self.fuse_outGlobal(globalCompressed), globalInput)
        interInput = torch.cat([tensor for tensor in (bba, bbt, bbv) if tensor is not None])
        interCompressed = self.fuse_inInter(interInput)
        interLoss = self.criterion(self.fuse_outInter(interCompressed), interInput)
    
        # globalCompressed = self.fuse_inGlobal(torch.cat((a, t, v)))
        
        # globalLoss = self.criterion(self.fuse_outGlobal(globalCompressed), torch.cat((a, t, v)))
        # interCompressed = self.fuse_inInter(torch.cat((bba, bbt, bbv)))
        # interLoss = self.criterion(self.fuse_outInter(interCompressed), torch.cat((bba, bbt, bbv)))

        loss = globalLoss + interLoss

        return torch.cat((globalCompressed, interCompressed), 0), loss
    
    # def forward1(self, a=None, t=None, v=None):
    #     B = self.projectB(torch.ones(460))
    #     if a is not None:
    #         A = self.projectA(a)
    #         BA = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), A), dim=1)
    #         bba = torch.mm(BA, torch.unsqueeze(A, dim=1)).squeeze(1)
    #     else: 
    #         bba= None
    #     if t is not None:
    #         T = self.projectT(t)
    #         BT = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), T), dim=1)
    #         bbt = torch.mm(BT, torch.unsqueeze(T, dim=1)).squeeze(1)
    #     else: 
    #         bbt= None
    #     if v is not None:
    #         V = self.projectV(v)
    #         BV = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), V), dim=1)
    #         bbv = torch.mm(BV, torch.unsqueeze(V, dim=1)).squeeze(1)
    #     else: 
    #         bbv= None

    #     # globalCompressed = self.fuse_inGlobal(torch.cat((a, t, v)))
    #     globalInput= torch.cat([tensor for tensor in (a,t,v) if tensor is not None])
    #     globalCompressed = self.fuse_inGlobal(globalInput) 
    #     globalLoss = self.criterion(self.fuse_outGlobal(globalCompressed), globalInput)
    #     interInput = torch.cat([tensor for tensor in (bba, bbt, bbv) if tensor is not None], dim=0)
    #     interCompressed = self.fuse_inInter(interInput)
    #     interLoss = self.criterion(self.fuse_outInter(interCompressed), interInput)
    
    #     # globalLoss = self.criterion(self.fuse_outGlobal(globalCompressed), torch.cat((a, t, v)))
    #     # interCompressed = self.fuse_inInter(torch.cat((bba, bbt, bbv)))
    #     # interLoss = self.criterion(self.fuse_outInter(interCompressed), torch.cat((bba, bbt, bbv)))

    #     loss = globalLoss + interLoss

    #     return torch.cat((globalCompressed, interCompressed), 0), loss