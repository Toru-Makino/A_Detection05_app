# ネットワーク定義用ファイル

import torch
import torch.optim as optim
import torch.nn.functional as F
#Pytorchの画像処理
import torchvision
# このモジュールは、画像の前処理（リサイズ、正規化、データ拡張など）を提供。
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torch.nn as nn


# データセット関数の定義
class Custom_Dataset(Dataset):
  def __init__(self,img_list):
    self.img_list = img_list

    # 画像の前処理として、リサイズとテンソルへの変換を定義
    self.prepocess = T.Compose([T.Resize((128,128)),
                                T.ToTensor(),
                                ])

  # 特定のインデックス idx のデータを返すメソッド。 指定されたインデックスの画像を開き、前述の前処理を適用。
  def __getitem__(self,idx):
    img = Image.open(self.img_list[idx])
    img = self.prepocess(img)
    return img

  # データセットの総数を返すメソッド。 画像リストの長さを返す。
  def __len__(self):
    return len(self.img_list)



# AutoEncoderの構築

class CustomModel(nn.Module):
    def __init__(self):

        # super関数は、親クラスのメソッドを呼び出す。nn.Moduleクラスの初期化メソッドを呼び出し。
        super(CustomModel,self).__init__()
        # Encoderの構築。
        # nn.Sequential内にはEncoder内で行う一連の処理を記載(コンテナ)する。入力が各層を順番に通過。
        # create_convblockは複数回行う畳み込み処理をまとめた関数。
        # 畳み込み層（create_convblockメソッドで定義）とプーリング層（nn.MaxPool2d）の組み合わせを使用。
        self.Encoder = nn.Sequential(self.create_convblock(3,16),     #256
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(16,32),    #128
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(32,64),    #64
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(64,128),   #32
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(128,256),  #16
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(256,512),  #8
                                    )
        # Decoderの構築。
        # ここでは逆畳み込み層（create_deconvblockメソッドで定義）と畳み込み層の組み合わせを使用
        self.Decoder = nn.Sequential(self.create_deconvblock(512,256), #16
                                     self.create_convblock(256,256),
                                     self.create_deconvblock(256,128), #32
                                     self.create_convblock(128,128),
                                     self.create_deconvblock(128,64),  #64
                                     self.create_convblock(64,64),
                                     self.create_deconvblock(64,32),   #128
                                     self.create_convblock(32,32),
                                     self.create_deconvblock(32,16),   #256
                                     self.create_convblock(16,16),
                                    )
        # 最後の出力を調整するための畳み込み層を定義。
        self.last_layer = nn.Conv2d(16,3,1,1)

    # 畳み込み層ブロックを作成するメソッドを定義。畳み込み層、バッチ正規化、ReLU活性化関数の組み合わせを2回繰り返す。
    def create_convblock(self,i_fn,o_fn):
        conv_block = nn.Sequential(nn.Conv2d(i_fn,o_fn,3,1,1),
                                   nn.BatchNorm2d(o_fn),
                                   nn.ReLU(),
                                   nn.Conv2d(o_fn,o_fn,3,1,1),
                                   nn.BatchNorm2d(o_fn),
                                   nn.ReLU()
                                  )
        return conv_block

    # 逆畳み込みブロックを作成するメソッドを定義。逆畳み込み（アップサンプリング）、バッチ正規化、ReLU活性化関数から構成。
    def create_deconvblock(self,i_fn , o_fn):
        deconv_block = nn.Sequential(nn.ConvTranspose2d(i_fn, o_fn, kernel_size=2, stride=2),
                                      nn.BatchNorm2d(o_fn),
                                      nn.ReLU(),
                                     )
        return deconv_block

    # 順伝播を定義。入力xがエンコーダを通過し、次にデコーダを通過し、最後に調整層を通過。この出力がモデルの最終出力。
    def forward(self,x):
        x = self.Encoder(x) #潜在変数はEncoderの出力に当たる為、これが潜在変数 of AutoEncoder (Encoder→潜在変数→Decoder)
        x = self.Decoder(x)
        x = self.last_layer(x)
        return x
    