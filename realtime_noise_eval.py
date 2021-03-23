import os
import cv2
import sys
import yaml
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool

##################### model stuff #####################
# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.utils import colorEncode
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger

sys.path.insert(1, '/home/zyang/Documents/Noisey-image')
#from noise_video_gen import *
from noises import *

# pass in mode config(yaml file)
# return a dict for the file 
# return decoder and encoder weights path
def parse_model_config(path):
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    encoder_path = None
    decoder_path = None
    for p in os.listdir(data['DIR']):
        if "encoder" in p.lower():
            encoder_path = "{}/{}".format(data['DIR'], p)
            continue
        if "decoder" in p.lower():
            decoder_path = "{}/{}".format(data['DIR'], p)
            continue

    if encoder_path==None or decoder_path==None:
        raise("model weights not found")
        
    return data, encoder_path, decoder_path

def visualize_result(img, pred, index=None, show=False):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    if show==True:
        display(PIL.Image.fromarray(im_vis))
    else:
        return pred_color, im_vis

def process_img(path=None, frame=None):
    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    # pil_image = PIL.Image.open('../ADE_val_00001519.jpg').convert('RGB')
    if path!=None:
        pil_image = PIL.Image.open(path).convert('RGB')
    else:
        pil_image = PIL.Image.fromarray(frame)

    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    return img_original, singleton_batch, output_size

def predict_img(segmentation_module, singleton_batch, output_size):
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred


def get_color_palette(pred, bar_height):

    pred = np.int32(pred)
    pixs = pred.size

    top_left_y = 0
    bottom_right_y = 30
    uniques, counts = np.unique(pred, return_counts=True)

    # Create a black image
    # bar_height = im_vis.shape[0]
    img = np.zeros((bar_height,250,3), np.uint8)

    for idx in np.argsort(counts)[::-1]:
        color_index = uniques[idx]
        name = names[color_index + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            #print("{}  {}: {:.2f}% {}".format(color_index+1, name, ratio, colors[color_index]))
            img = cv2.rectangle(img, (0,top_left_y), (250,bottom_right_y), 
                       (int(colors[color_index][0]),int(colors[color_index][1]),int(colors[color_index][2])), -1)
            img = cv2.putText(img, "{}: {:.3f}%".format(name, ratio), (0,top_left_y+20), 5, 1, (255,255,255), 2, cv2.LINE_AA)
            top_left_y+=30
            bottom_right_y+=30
            
    return img


def transparent_overlays(image, annotation, alpha=0.5):
    img1 = image.copy()
    img2 = annotation.copy()

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg, img2_fg)
    dst = cv2.addWeighted(image.copy(), 1-alpha, img2_fg, alpha, 0)
    img1[0:rows, 0:cols ] = dst
    return dst


def load_model_from_cfg(cfg):
    model_config, encoder_path, decoder_path = parse_model_config(cfg)
    net_encoder = ModelBuilder.build_encoder(
        arch = model_config["MODEL"]['arch_encoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        weights = encoder_path)
    net_decoder = ModelBuilder.build_decoder(
        arch = model_config["MODEL"]['arch_decoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        num_class = model_config['DATASET']['num_class'],
        weights = decoder_path,
        use_softmax=True)
    
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    return segmentation_module

##################### eval functions #####################
def get_anno(anno_path):
    anno = PIL.Image.open(anno_path)
    anno = np.array(anno)
    anno[np.where(anno!=0)]-=1
    return anno

def get_eval_res(pred, anno):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    acc, pix = accuracy(pred, anno)
    intersection, union = intersectionAndUnion(pred, anno, 150)
    acc_meter.update(acc, pix)
    intersection_meter.update(intersection)
    union_meter.update(union)
    
    class_ious = {}
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        class_ious[i] = _iou
    return iou.mean(), acc_meter.average(), class_ious


# return class_ious!=0
# only count for classes that appeared in anno
# it is able to update the pass in dict
def process_class_ious(class_ious, anno, name_iou=None):
    gt_classes = np.unique(anno)
    tmp_dict = {k: v for k, v in sorted(class_ious.items(), key=lambda item: item[1], reverse=True) if v!=0}
    
    if name_iou==None:
        name_iou = {}
    for k, v in tmp_dict.items():
        if k in gt_classes:
            try:
                name_iou[names[k+1]].append(v)
            except:
                name_iou[names[k+1]] = [v]
    return name_iou


def sub_line_plt(ax, x, y, title=None, label=None, color='r', yl=None, xl=None, clear=True, ylim=None):
    if clear:
        ax.clear()
    ax.set_title(title)
    
    if ylim!=None:
        ax.set_ylim(ylim)
    
    if color!='random':
        ax.plot(x, y, label=label, color=color)
    else:
        ax.plot(x, y, label=label)
    ax.set_ylabel(yl)
    ax.set_xlabel(xl)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Predict on image")
    parser.add_argument("-i", "--img", required=True, type=str, metavar='', help="an image path")
    parser.add_argument("-a", "--alpha", default=0.6, type=float, metavar='', help="transparent overlay level")
    parser.add_argument("-s", "--save", default="tmp_results/", type=str, metavar='', help="save prediction to")
    parser.add_argument("-d", "--display", default=1, type=int, metavar='', help="display real time prediction")
    
    parser.add_argument("--cfg", default="config/ade20k-resnet50dilated-ppm_deepsup.yaml", 
                        metavar="FILE", help="path to config file", type=str,)
    parser.add_argument("--gpu", default=0, type=int, metavar='', help="gpu id for evaluation")
    parser.add_argument("opts", help="Modify config options using the command-line", 
                        default=None, nargs=argparse.REMAINDER, metavar='')
    
    parser.add_argument("--anno", default="/home/zyang/Documents/fork_sseg_mit/notebooks/data/ADEChallengeData2016/annotations/training/ADE_train_00000001.png", type=str, metavar='', help="path to an annotation")
    
    parser.add_argument("--figsize", default="13,12", type=str, metavar='', help="size of the figure")
    parser.add_argument("-l", "--legend", default=1, type=int, metavar='', help="display legend or not")
    parser.add_argument("--rate", default=0.0001, type=float, metavar='', help="noise level increase rate")
    # 0: org | pred | cp
    # 1: org Vertical pred | cp
    # 2: org | pred
    # 3: org Vertical pred
    parser.add_argument("-dm", "--dmode", default=0, type=int, metavar='', help="display mode")
    parser.add_argument("-r", "--ratio", default=1, type=float, metavar='', help="Ratio for resizing frame")
    
    args = parser.parse_args()
    
    mode = args.dmode
    r = args.ratio
    
    # colors
    colors = scipy.io.loadmat('data/color150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    
    # Network Builders
    print("parsing {}".format(args.cfg))
    segmentation_module = load_model_from_cfg(args.cfg)
    segmentation_module.eval()
    segmentation_module.cuda()
    
    # For eval 
    accs = []
    mean_ious = []
    class_num = []
    className_ious = {}
    
    noise_levels = []
    
    # read img for pred
    # read annotation img
    img = cv2.imread(args.img)
    org_img = img.copy()
    
    amount = 0.0
    #rate = 0.0001
    rate = args.rate
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2,2, figsize=(int(args.figsize.split(',')[0]), int(args.figsize.split(',')[1])))
    
    # it can take a anno OR compare to itself
    self_compare = 0
    try:
        anno = get_anno(args.anno)
        gt_classes = np.unique(anno)
    except:
        self_compare = 1
        img_original, singleton_batch, output_size = process_img(frame=img)        
        anno = predict_img(segmentation_module, singleton_batch, output_size)
        gt_classes = np.unique(anno)
        fig.suptitle('NO Annotation', fontsize=15)
        
    
    # test Process
    p = Pool(processes=2)
    
    while(True):
        
        if self_compare==1:
            amount = amount + rate
            self_compare = 0
            continue
        
        noise_levels.append(amount)
        img_original, singleton_batch, output_size = process_img(frame=img)        
        pred = predict_img(segmentation_module, singleton_batch, output_size)
        
        mean_iou, acc, class_ious = get_eval_res(pred, anno)
        className_ious = process_class_ious(class_ious, anno, className_ious)
        
        accs.append(acc)
        mean_ious.append(mean_iou)
        tmp_class_num = np.unique(pred)
        class_num.append(len(set(gt_classes) & set(tmp_class_num)))        
        
        # realtime plotting 
        sub_line_plt(axs[0][0], noise_levels, accs, title='Pixel Accuracy', 
                     label="acc", color='r', yl='Pixel Accuracy', xl=None, ylim=[0,1])
        
        sub_line_plt(axs[0][1], noise_levels, mean_ious, title='Mean IoU', 
                     label="meanIoU", color='r', yl='Mean IoU', xl=None, ylim=[0,max(mean_ious)])
        
        sub_line_plt(axs[1][0], noise_levels, class_num, label="class_num", title='Number of Classes',
                     color='b', yl='Number of Classes', xl='Noise Levels (Amount)', ylim=[0, max(class_num)])
        
        axs[1][1].clear()
        for k, v in className_ious.items():
            sub_line_plt(axs[1][1], noise_levels[:len(v)], v, label=k, title='Class IoU', color='random',
                     yl='Class IoU', xl='Noise Levels (Amount)', clear=False, ylim=[0,1])
            
        if args.legend==1:
            axs[1][1].legend()
        
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        fig.tight_layout()
        plt.pause(0.005)
        
        # Multi process on noise function
        #img = saltAndPapper_noise(img, amount)
        probs = partial(saltAndPapper_noise, prob=amount)
        data = p.map(probs, [img])
        img = data[0]
        
        amount = amount + rate
        
        pred_color, org_pred_split = visualize_result(img_original, pred)
        
        # split org | pred | cp
        if mode==0:
            #color_palette = get_color_palette(pred, org_pred_split.shape[0])
            #frame = numpy.concatenate((org_pred_split, color_palette), axis=1)
            frame = numpy.concatenate((img_original, org_img), axis=1)
            frame = numpy.concatenate((frame, pred_color), axis=1)
        # split noise |v org |v pred | cp
        elif mode==1:
            # frame = numpy.concatenate((img_original, pred_color), axis=0)
            frame = numpy.concatenate((img_original, org_img), axis=0)
            frame = numpy.concatenate((frame, pred_color), axis=0)
            #color_palette = get_color_palette(pred, frame.shape[0])
            #frame = numpy.concatenate((frame, color_palette), axis=1)
        '''
        elif mode==2:
            #frame = org_pred_split
            frame = numpy.concatenate((img_original, org_img), axis=1)
            frame = numpy.concatenate((frame, pred_color), axis=1)
            color_palette = get_color_palette(pred, frame.shape[0])
            frame = numpy.concatenate((frame, color_palette), axis=1)
        elif mode==3:
            #frame = numpy.concatenate((img_original, pred_color), axis=0)
            frame = numpy.concatenate((img_original, org_img), axis=0)
            frame = numpy.concatenate((frame, pred_color), axis=0)
            color_palette = get_color_palette(pred, frame.shape[0])
            frame = numpy.concatenate((frame, color_palette), axis=1)
        '''
        
        if (args.display)==1:
            dsize = (int(r*frame.shape[1]), int(r*frame.shape[0]))
            frame = cv2.resize(frame, dsize)
            cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break          
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)
         
    
    p.close()
    cv2.destroyAllWindows()
    plt.show()
    
    
    '''        
    # predict
    img_original, singleton_batch, output_size = process_img(args.img)
    pred = predict_img(segmentation_module, singleton_batch, output_size)
    # print(type(img_original))
    pred_color, org_pred_split = visualize_result(img_original, pred)
    
    # color_palette
    color_palette = get_color_palette(pred, org_pred_split.shape[0])
    
    # transparent pred on org
    dst = transparent_overlays(img_original, pred_color, alpha=args.alpha)
    
    # colored_pred + color_palette
    pred_color_palette = numpy.concatenate((color_palette, pred_color), axis=1)
    
    # transparent pred on org + color_palette
    pred_color_palette_dst = numpy.concatenate((color_palette, dst), axis=1)
    
    # org + colored_pred + color_palette
    pred_color_palette_all = numpy.concatenate((org_pred_split, color_palette), axis=1)
    
    cv2.imwrite("{}/pred_color.png".format(args.save), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/org_pred_split.png".format(args.save), cv2.cvtColor(org_pred_split, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/dst.png".format(args.save), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette.png".format(args.save), cv2.cvtColor(pred_color_palette, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette_dst.png".format(args.save), cv2.cvtColor(pred_color_palette_dst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette_all.png".format(args.save), cv2.cvtColor(pred_color_palette_all, cv2.COLOR_RGB2BGR))
    
    if (args.display)==1:
        PIL.Image.fromarray(pred_color_palette_dst).show()
    else:
        print("results saved")
    '''