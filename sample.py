import cv2

def detect_human(image_path):
    # 顔認識用の学習済み分類器を読み込む
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 画像を読み込む
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔を検出する
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 顔が検出された場合は〇、検出されなかった場合は×を返す
    if len(faces) > 0:
        return "〇"
    else:
        return "×"

# 画像のパス
image_path = "d8710952b8f951c661174ec48194f787.jpg"

# 人間の識別を実行
result = detect_human(image_path)
print("人間の識別結果:", result)


