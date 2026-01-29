import sys
import os
import json
import time

import torch

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()
        
        losses = model(image, target)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    # print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, epoch, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, epoch, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    # print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
    metrics = {}
    
    for iou_type in iou_types:
        stats = coco_evaluator.coco_eval[iou_type].stats  # 12-element array
        if stats is not None and len(stats) == 12:
            metrics[iou_type] = {
                "AP@[IoU=0.50:0.95]": stats[0],
                "AP@0.50": stats[1],
                "AP@0.75": stats[2],
                "AP_small": stats[3],
                "AP_medium": stats[4],
                "AP_large": stats[5],
                "AR@1": stats[6],
                "AR@10": stats[7],
                "AR@100": stats[8],
                "AR_small": stats[9],
                "AR_medium": stats[10],
                "AR_large": stats[11],
            }
        else:
            metrics = None
    return output, iter_eval, metrics
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, epoch, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    # print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
    
    # Load old results if file exists
    json_path = os.path.splitext(args.results)[0] + ".json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results_data = json.load(f)
    else:
        results_data = []

    # Append this epoch's results
    results_data.append({"epoch": epoch, "results": coco_results})

    # Save back
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=4)
        
    return A / iters
    

