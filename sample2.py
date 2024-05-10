import cv2

def detect_human_dnn(image_path):
    # 顔検出モデルの設定
    model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_path = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 画像を読み込めませんでした。ファイルパスを確認してください。")
        return

    # 画像の高さと幅を取得
    (h, w) = image.shape[:2]

    # 画像を前処理する
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # モデルに画像を入力し、顔を検出する
    net.setInput(blob)
    detections = net.forward()

    # 検出された顔を矩形で囲む
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 信頼度が0.5以上の検出結果を使用
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 顔が検出された画像を表示する
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像のパスをユーザーに入力してもらう
image_path = input("画像のパスを入力してください: ")

# 人間の識別を実行
detect_human_dnn(image_path)

