import os
import cv2
import glob


def video2imgs(video_file: str,
               frame_mabiki: int = 1,
               debug: bool = False):
    # out-dir
    _name = video_file.split('.')[0]
    out_dir = f'./{_name}'
    os.makedirs(out_dir, exist_ok=True)

    # video
    video = cv2.VideoCapture(video_file)

    if debug:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        print('frame_size: ', width, height)
        print('frame_num: ', frame_num)
        print('frame_rate: ', frame_rate)

    # read frames
    frame_cnt = 0
    while True:
        ret, frame = video.read()
        if ret == False:
            break

        if frame_cnt % int(frame_mabiki) == 0:
            cv2.imwrite(f'{out_dir}/{frame_cnt:05d}.jpg', frame)

        frame_cnt += 1

    video.release()


def imgs2video(img_dir: str,
               video_file: str = 'video.mp4',
               frame_rate: float = 1.0,
               resize_ratio: float = 1.0,
               frame_mabiki: int = 1):
    # check_img_size
    _img_file = glob.glob(f'{img_dir}/*.jpg')[0]
    _img = cv2.imread(_img_file, cv2.IMREAD_GRAYSCALE)
    _h = _img.shape[0]
    _w = _img.shape[1]
    size = (int(_w * resize_ratio), int(_h * resize_ratio))

    # video
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_file, fmt, frame_rate, size)

    for img_cnt, img_path in enumerate(glob.glob(f'{img_dir}/*.jpg')):
        if img_cnt % int(frame_mabiki) == 0:
            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, size)
            writer.write(frame)

    writer.release()


if __name__ == '__main__':
    video2imgs('IMG_2327.MOV', debug=True)
    # imgs2video(img_dir='./IMG_2327')
