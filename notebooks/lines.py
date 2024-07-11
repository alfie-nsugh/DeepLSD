import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
import h5py

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = {
    'detect_lines': True,  # Whether to detect lines or only DF/AF
    'line_detection_params': {
        'merge': True,  # Whether to merge close-by lines
        'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        'grad_thresh': 1.3,
        'grad_nfa': False,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    }
}
# returns model
def load_model(model_name="md", conf=conf):
    if model_name == "md":
        chkpnt = "../weights/deeplsd_md.tar"
    elif  model_name == "wireframe":
        chkpnt = "../weights/deeplsd_wireframe.tar"
    chkpnt = torch.load(chkpnt, map_location=torch.device('cpu'))
    net = DeepLSD(conf)
    net.load_state_dict(chkpnt['model'])
    net = net.to(device).eval()
    return net


def filter_lines(lines, min_length=10):
    """
    Filter out short lines
    """
    filtered_lst =  [line for line in lines if np.linalg.norm(line[0] - line[1]) >= min_length]
    return np.array(filtered_lst)


def get_square(line_arr, corner_pt, thresholds = [5]):
    # get the distance from corners for each line
    dist = np.sqrt(np.sum((line_arr.reshape(-1,2) - corner_pt)**2, axis=1))
    # get the index of the closest line and second closest line
    partitioned_indices = np.argpartition(dist, 2)[:2]
    closest = partitioned_indices[0]
    second_closest = partitioned_indices[1]

    # get the actual lines using unravel index
    closest_line = np.unravel_index(closest, line_arr.shape[:-1])
    second_closest_line = np.unravel_index(second_closest, line_arr.shape[:-1])

    # get the actual lines
    closest_line = line_arr[closest_line[0],:]
    second_closest_line = line_arr[second_closest_line[0],:]

    square_lst = []
    # put both in a queue
    queue = [closest_line, second_closest_line]
    
    line_arr = np.delete(line_arr, np.where((line_arr == closest_line).all(axis=1)), axis=0)
    line_arr = np.delete(line_arr, np.where((line_arr == second_closest_line).all(axis=1)), axis=0)

    while len(queue):
        # get the first element
        line = queue.pop(0)
        square_lst.append(line)

        #plot_images([img], ['DeepLSD lines'], cmaps='gray')
        #plot_lines([np.array(square_lst)], line_colors='green', indices=range(1))
        
        # save fig to file
        #plt.savefig('square.png')

        # delete the line from the array
        line_arr = np.delete(line_arr, np.where((line_arr == line).all(axis=1)), axis=0)
        # get the distance from line to line_arr
        dist1 = np.sqrt(np.sum((line_arr.reshape(-1,2) - line[0])**2, axis=1))
        dist2 = np.sqrt(np.sum((line_arr.reshape(-1,2) - line[1])**2, axis=1))
        # get the indices of the lines that are close to the line

        #indices = np.where(dist < thresholds[0])
        smallest_dist_index1 = np.argmin(dist1)
        smallest_dist_index2 = np.argmin(dist2)
        # check if the distance is small enough

        dist = dist1 if dist1[smallest_dist_index1] < dist2[smallest_dist_index2] else dist2
        smallest_dist_index = smallest_dist_index1 if dist1[smallest_dist_index1] < dist2[smallest_dist_index2] else smallest_dist_index2

        #print("orig line: ", line, dist1[smallest_dist_index1], dist2[smallest_dist_index2], dist[smallest_dist_index])

        if dist[smallest_dist_index] < thresholds[0]:
            # get the line that is closest
            line_index = np.unravel_index(smallest_dist_index, line_arr.shape[:-1])
            line = line_arr[line_index[0],:]
            # put the line in the queue
            queue.append(line)
        
    return np.array(square_lst)

def load_images_in_folder(images_path):
    images = os.listdir(images_path)
    # Filter out non-image files
    image_names = [image.split("/")[-1] for image in images if image.endswith('.jpg')]
    # Load images
    images = [cv2.imread(os.path.join(images_path, image_name))[:, :, ::-1] for image_name in image_names]

    print(image_names)

    return images, image_names

def change_resolution(images, width, height):
    for i, image in enumerate(images):
        images[i] = cv2.resize(image, (width, height))

    return images

def get_lines(model, images):
    lines_lst = []
    res_lst = []
    for i, image in enumerate(images):
        if image.shape[0] > 2732 or image.shape[1] > 1820:
            image = change_resolution([image], image.shape[1]//2, image.shape[0]//2)[0]
            # TODO make generic, remove hardcoded 1820, 2732
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        input = {'image' : torch.tensor(gray_image, dtype=torch.float, device=device)[None,None]/255.}
        with torch.no_grad():
            output = model(input)
            lines = output['lines'][0]
        lines_lst.append(lines)
        res_lst.append(gray_image.shape[:2])
        print("image: ", i)

    return lines_lst, res_lst






if __name__ == "__main__":
    imgpath = "../../sam3d/data/crops/1DJI0001"
    images, image_names = load_images_in_folder(imgpath)
    model = load_model("md")
    lines_lst, res_lst = get_lines(model, images)
    
    filtered_lines = []
    for l,res in zip(lines_lst, res_lst):
        threshold = min(res[0],res[1])/10
        filtered_lines.append(filter_lines(l, threshold))

    # check if images folder exists if not create
    if not os.path.exists('images'):
        os.makedirs('images')

    # save images + line to image
    for i, (img, lines) in enumerate(zip(images, filtered_lines)):
        if (img.shape[0] > 2732 or img.shape[1] > 1820):
            img = change_resolution([img], img.shape[1] // 2, img.shape[0] //2)[0]
        plot_images([img], ['DeepLSD lines'], cmaps='gray')
        plot_lines([lines], line_colors='green', indices=range(1))
        plt.savefig(f'images/{image_names[i]}')

    # save lines to pickle
    import pickle
    with open('lines.pkl', 'wb') as f:
        pickle.dump(filtered_lines, f)
    with open('res.pkl', 'wb') as f:
        pickle.dump(res_lst, f)