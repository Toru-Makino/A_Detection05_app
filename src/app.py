# View ファイル　学習済みモデルを組み込んで、実際に動作させる。
# predict()関数の中の推論
# アップロードされた画像に対する処理にする事

import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np

# このモジュールは、画像の前処理（リサイズ、正規化、データ拡張など）を提供。
import torchvision.transforms as T

from flask import Flask, request, render_template, redirect
from flask import url_for
import io
from PIL import Image
import base64
import uuid
import os

# a_detection.py から前処理とネットワークの定義を読み込み
from a_detection import CustomModel


### 異常検知専用コード (始) ###


# 画像の前処理を行うための変換関数を定義。画像を128x128にリサイズし、PyTorchのテンソルに変換、の2つの前処理を行う。
prepocess = T.Compose([T.Resize((128,128)),
                                T.ToTensor(),
                                ])

def process_image(img):

    # ネットワークの準備
    model = CustomModel().cpu().eval()

    # 学習済みモデルの重み ( model.pt ) を読み込み

    # Visual Studio Code の際のコードは以下の通り
    # model.load_state_dict(torch.load('./src/model_t.pt', map_location=torch.device('cpu')))

    # Render にデプロイする際のコードはコメントアウト
    model.load_state_dict(torch.load('./model_t.pt', map_location=torch.device('cpu')))

    # 画像の余白幅をピクセル単位で指定。これは後で画像を連結する際に使用。
    margin_w = 10

    # 先ほど定義した前処理を適用し、次元を増やしてバッチ次元を追加（.unsqueeze(0)）。
    # unsqueeze(0)でバッチ次元を追加、1つの画像テンソルを1つのバッチとして扱えるようになる。
    # モデルにデータをバッチとして入力するために必要。
    # バッチ次元追加でテンソルの形状は(1, C, H, W)になる。ここで、Cはチャネル数、Hは高さ、Wは幅を表す。    
    img = prepocess(img).unsqueeze(0)

    # このコードブロック内では、PyTorchが勾配を計算するのを防ぎ、メモリ使用量を減らし、評価フェーズの計算を高速化。
    with torch.no_grad():

        # モデルに画像を通して結果を取得。結果はテンソルのリストとして返されるため、最初の要素を取得。
        # [0]を使うことで、Autoencoderによる処理を経た後の最初のテンソル（再構築された出力テンソル）が取得されます。
        output = model(img)[0]

    # Numpy配列に変換、さらに軸を変換（チャネル次元を最後に移動）。    
    output = output.cpu().numpy().transpose(1,2,0)    

    # output * 255: outputは0から1の範囲にスケーリングされた浮動小数点数の値を持つ多次元配列。この処理により値が0から255の範囲にスケーリングされる。
    # np.minimum(output * 255, 255): output * 255の結果と255との要素ごとの最小値を計算。これにより、値が255を超える場合には255にクリップされます。つまり、255を上限として値がクリップされる。
    # np.maximum(np.minimum(output * 255, 255), 0): 先ほどクリップした値と0との要素ごとの最大値を計算。これにより、値が0未満の場合には0にクリップ。つまり、0を下限として値がクリップされる。
    # np.uint8(...): 最後に、値を整数化。np.uint8はNumPyのデータ型の1つで、8ビット符号なし整数型。この操作により、値が小数点以下を持たない0から255の整数に変換。
    output = np.uint8(np.maximum(np.minimum(output*255 ,255),0))

    # Autoencoderを通す前の元画像の処理。
    # img[0]: imgはPyTorchのテンソルで、バッチ次元を持つ。img[0]でバッチ次元を削除、最初の1つの画像データを取得。imgの形状が(1, C, H, W)から(C, H, W)に変更。
    # * 255: テンソルの値を255倍。0から1の範囲の浮動小数点数が0から255の整数にスケーリングされる。
    # 最終的に得られるのは、画像を0から255の整数値として表現されるNumpy多次元配列。画像データ操作・可視化に便利。
    origin = np.uint8(img[0].cpu().numpy().transpose(1,2,0)*255)

    # output画像とorigin画像の絶対値の差分をdffに格納。
    # astype(np.float32)で、outputのデータ型を32ビット浮動小数点数に変換。後の計算で精度の高い演算が保証される。
    # np.abs()関数は、要素ごとに絶対値を計算するため、差の絶対値が取得できる。
    # np.uint8(...): 最後に、値を整数化。
    diff = np.uint8(np.abs(output.astype(np.float32) - origin.astype(np.float32)))

    # OpenCVのapplyColorMap関数を使って、差分画像(diff)をカラーマップに適用、色付きのヒートマップを作成。
    # cv2.applyColorMap()は、グレースケールの画像にカラーマップを適用してカラー画像を生成する関数。
    # 第一引数には2つの画像間の差異を表すグレースケール画像 diffを指定、第2引数にはカラーマップの種類を指定。
    # cv2.COLORMAP_JETは、OpenCVが提供するカラーマップの一つで、青から赤までの色で値を表現。
    # ヒートマップは、画像の差異を視覚的に表現し、特にデータの密度やパターンを強調するために使用。 
    heatmap = cv2.applyColorMap(diff , cv2.COLORMAP_JET)

    # diffはNumPyの多次元配列であり、画像の高さ（行数）を表す次元の長さがdiff.shape[0]。
    # margin_wは整数値 (10) であり、マージンの幅を指定。マージンは画像の周囲に追加される空白の幅。
    # np.ones()関数で全ての要素が1の多次元配列を生成。(diff.shape[0], margin_w, 3)の形状の多次元配列を作成。3はカラーチャンネルの数（RGB）。
    # * 255: 上記の多次元配列にスカラー値255を乗算、全ての要素が255（白色）になる。
    # diffと同じ高さとマージンの幅を持つ白いカラー画像（マージン）を生成。画像間に白い余白を作成して区別しやすくする。
    margin = np.ones((diff.shape[0],margin_w,3))*255
    
    # 3つのNumPyの多次元配列（origin、margin、output、heatmap）を横に連結して1つのカラー画像（result）を作成。
    # origin & output はNumPyの多次元配列であり、RGB形式のカラー画像。RGBチャンネルの順序を逆転させることにより、青、緑、赤の順で並べ替えられたカラー画像に変換。
    # np.concatenate()関数を使って、上記の4つの多次元配列を横に連結して1つのカラー画像に結合。axis=1は、連結の方向を示し、1は横方向（列方向）。
    result = np.concatenate([origin[:,:,::-1],margin,output[:,:,::-1],margin,heatmap],axis = 1)

    # 異常検知専用コードで得られた result を、image に入れ替えて、強制的にキカガクのコードに接続。
    result_scaled = result.astype(np.uint8)
    result_rgb = result_scaled[..., [2, 1, 0]]
    image = Image.fromarray(result_rgb)

    return image

### 異常検知専用コード (終) ###




# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合に、POSTリクエストされた場合に、predict関数が呼び出されて実行される。
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # index.htmlにアップロードされた filename と言うファイルデータを変数fileに代入。
        file = request.files['filename']
        # ファイルが存在し、指定されたファイル形式である場合、以下を実行。
        if file and allowed_file(file.filename):

            # Pythonの io.BytesIO を使って、バイト型のデータを一時的に保持するためのバッファ（一種の一時記憶領域）を作成。
            # 画像を一時的に保存し、その後の処理に使用するため。
            buf = io.BytesIO()

            # PILを使用してアップロードされた画像を開き、.convert('RGB') で画像をRGB形式に変換。
            img = Image.open(file).convert('RGB')



            # 初期設定の AutoEncoder 関数呼び出し
            image = process_image(img)            



            #　前で作成したバッファ buf に開いた画像をPNG形式で保存。
            image.save(buf, 'png')

            # バッファに保存した画像データを base64 形式でエンコードし、それを utf-8 でデコード。
            # base64 形式はバイナリデータを文字列に変換する一般的な方法で、
            # バイナリデータをテキストベースのメディア（例えばHTML）で安全に送受信するためによく使われる。
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')

            # 画像をHTMLで表示するための data: URLを作成している。
            # これにより、result.html の <img> タグの src 属性に直接この文字列を設定できる。
            # result.html では {{ image }} とあり、image ファイルは前述の画像ファイルであり、
            # 最後の return render。。。に登場する、image=base64_data。
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # キカガクでは以下のコードが入る    
            # 入力された画像に対して推論
            # pred = predict(image)
            # animalName_ = getName(pred)

            # 最後に、render_template を用いて結果を表示するHTMLページ（'result.html'）をレンダリング（生成）。
            # その際、テンプレートに animalName_ と base64_data の2つの変数を渡している。
            # これらの変数は、'result.html'の中で使われ、動物の名前と画像を表示する。
            return render_template('result.html', image=base64_data)




    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
