import cv2
from pathlib import Path

def get_count(capture):
    # Try to get most last image in folder, so we don't overwrite images
    try:
        last = sorted(os.listdir(f'data/{capture}'))[-1]
        count = int(last[:-4])+1
    # Otherwise set count to 
    except:
        count = 0

    return count


def main():
    capture = 'good'
    record  = False
    cv2.namedWindow("saving")

    # make all folders
    Path(f'data/good').mkdir(parents=True, exist_ok=True)
    Path(f'data/bad').mkdir(parents=True, exist_ok=True)
    Path(f'data/ugly').mkdir(parents=True, exist_ok=True)
    count = get_count(capture)

    # TODO : Make sure it's getting the right camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()

        # TODO : Setup get oreo square out of image
        oreo = image





        # Save Image
        cv2.imshow('saving', oreo)
        if record:
            cv2.imwrite(f'data/{capture}/{count:06d}.jpg', oreo)
            count += 1

        # Process key presses
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

        if key == ord('g'):
            capture = 'good'
            count = get_count(capture)
            print("Switching to good folder")

        if key == ord('b'):
            capture = 'bad'
            count = get_count(capture)
            print("Switching to bad folder")

        if key == ord('u'):
            capture = 'ugly'
            count = get_count(capture)
            print("Switching to ugly folder")

        if key == ord('r'):
            record = True
            print("Starting to save")

        if key == ord('s'):
            record = False
            print("Stopping saving")

if __name__ == "__main__":
    main()