import cv2
import numpy as np  # NumPyを追加

def detect_faces(image_path):
    # 顔検出用のプリトレーニング済みモデルをロードする
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 画像を読み込めませんでした。ファイルパスを確認してください。")
        return

    # 画像の高さと幅を取得
    (h, w) = image.shape[:2]

    # 画像の前処理
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # ネットワークに画像を入力し、顔を検出する
    net.setInput(blob)
    detections = net.forward()

    # 検出された顔の数をカウントする
    num_faces = 0

    # 検出された顔を矩形で囲み、画像に描画する
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 信頼度が0.5以上の場合のみ処理
            num_faces += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # NumPyを使用する
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 検出された顔の数と結果を表示する
    print("検出された顔の数:", num_faces)
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像のパスをユーザーに入力してもらう
image_path = input("画像のパスを入力してください: ")

# 顔の検出を実行
detect_faces(image_path)

