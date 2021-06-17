# импортируем библиотеки
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os

# устанавливаем MTCNN иInceptionResnetV1
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)  # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Считываем данные из папки
dataset = datasets.ImageFolder('photos')  # путь на папку с фото
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # применяем имена


def collate_fn(x):
    return x[0]


loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = []  # список имен
embedding_list = []  # список эмбэдингов

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob > 0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

data = [embedding_list, name_list]
torch.save(data, 'data.pt')  # сохраняем данные в файл для дальнейшего использования

# используем веб-камеру
# загружаем файл для загрузки
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Камера куда-то делась =)")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.80:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # список зафиксированных дистанций

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # отмечаем минимальную разницу
                min_dist_idx = dist_list.index(min_dist)  # отмечаем индекс минимальной разницы
                name = name_list[min_dist_idx]  # отмечаем имя минимальной разницы

                box = boxes[i]

                original_frame = frame.copy()  # копия кадра для отрисовки

                if min_dist < 0.95:
                    frame = cv2.putText(frame, name + ' ' + str(min_dist), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 1, cv2.LINE_AA)

                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    cv2.imshow("IMG", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # Выход
        print('Похоже, вы нажали ESC')
        break

    elif k % 256 == 32:  # Пространство для добавления новых людей
        print('Пока что я вас еще не знаю, введите ваше имя:')
        name = input()

        # создаем директорию для нового человека
        if not os.path.exists('photos/' + name):
            os.mkdir('photos/' + name)

        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))

cam.release()
cv2.destroyAllWindows()
