import cv2

def detect_human(image_path):
    # 顔認識用の学習済み分類器を読み込む
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 画像を読み込めませんでした。ファイルパスを確認してください。")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔を検出する
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出された顔の数をカウントする
    num_faces = len(faces)

    # 顔が検出されたかどうかを判定する
    if num_faces > 0:
        print("人間の識別結果: 〇")
        print("検出された顔の数:", num_faces)
        for (x, y, w, h) in faces:
            # 検出された顔を矩形で囲む
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 顔が検出された画像を表示する
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("人間の識別結果: ×")
        # 顔が検出されなかった場合も画像を表示する
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 画像のパスをユーザーに入力してもらう
image_path = input("画像のパスを入力してください: ")

# 人間の識別を実行
detect_human(image_path)

