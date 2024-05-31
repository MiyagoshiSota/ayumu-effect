import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Webカメラからの入力用の設定。背景色をグレーに設定。
BG_COLOR = (0, 0, 0)  # gray

# コントラストと明るさの変更
alpha = 1.4  # コントラストの倍率（1より大きい値でコントラストが上がる）
beta = 10  # 明るさの調整値（正の値で明るくなる）

cap = cv2.VideoCapture(0)  # Webカメラからビデオキャプチャを開始

# MediapipeのSelfieSegmentationモデルを読み込む
with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
    bg_image = None  # 背景画像を初期化
    while cap.isOpened():  # カメラがオープンされている間ループ
        success, image = cap.read()  # カメラからフレームを読み込む
        if not success:  # フレームが正しく読み込まれなかった場合
            print("Ignoring empty camera frame.")  # メッセージを表示して
            continue  # 次のフレームの読み込みに移る

        # 映像を水平に反転し、BGR形式からRGB形式に変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 画像を処理のために書き込み禁止にする
        results = selfie_segmentation.process(
            image)  # SelfieSegmentationモデルで処理

        image.flags.writeable = True  # 画像を再び書き込み可能にする
        # 処理後の画像をグレースケールに変換
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # セグメンテーション結果に基づいて条件を作成
        condition = np.stack((results.segmentation_mask, ) * 3, axis=-1) > 0.3
        if bg_image is None:  # 背景画像がまだ初期化されていない場合
            # グレースケール画像と同じ形状の3チャンネル画像を作成
            bg_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3),
                                dtype=np.uint8)
            bg_image[:] = BG_COLOR  # 背景画像をグレーに設定

        # グレースケール画像を再び3チャンネル画像に変換
        gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        adjusted_image = cv2.convertScaleAbs(gray_image_3ch,
                                             alpha=alpha,
                                             beta=beta)
        # セグメンテーションマスクに基づいて前景と背景を統合
        output_image = np.where(condition, adjusted_image, bg_image)

        # 結果の画像を表示
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        if cv2.waitKey(5) & 0xFF == 27:  # 'Esc'キーが押されたらループを終了
            break
    cap.release()  # カメラリソースを解放
cv2.destroyAllWindows()  # OpenCVのウィンドウをすべて閉じる
