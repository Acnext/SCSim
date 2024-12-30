import numpy as np
import argparse
import math
import scipy
import struct
import argparse
#from scipy import integrate
import cv2  # 利用opencv读取图像
import matplotlib.pyplot as plt 
from PIL import Image
import copy
import os
def get_spike(in_filepath = "D:\\lwhu\\gt\\airplane\\", out_filepath = "D:\\lwhu\\gt\\airplane\\", w = 400, h = 250,  class_id = 1, frame_start = 0, frame_end = 999, wins = 1):
    dark = 1
    #np.save(out_filepath + "dark.npy", dark)
    print(dark)
    lux = 5 * 1e4 * dark #模拟散粒噪声
    mx_dynamic_range = (1 << 8) - 1.0
    #img = np.zeros((h, w))
    accumulator = np.random.rand(h, w)
    byte = 0
    pos = 0
    Vrst = 3.3
    Vref = 2.2
    IE = 0.4442
    Is_sigma = 0.1731
    VE = -0.0076
    Vs_sigma = 0.0497
    CE = 4.348
    Cs_sigma = 0.0128
    AlphaE = 1.0128
    Alphas_sigma = 0.0584
    Vt_hot_sigma = 9.644 * 1e-9 #
    Vt_shot_sigma = 0.00636 #电流的散粒噪声
    noise_Idark = np.random.normal(IE, Is_sigma, (h, w))
    Cs = np.random.normal(CE, Cs_sigma, (h, w))
    Vs = np.random.normal(VE, Vs_sigma, (h, w))
    Alphas = np.random.normal(AlphaE, Alphas_sigma, (h, w)) / AlphaE #光电转化率
    #print(Alphas)
    is_no_noise = True#同时生成无噪数据
    It = np.zeros((h, w))
    Thres = np.zeros((h, w))
    is_delay = np.zeros((h, w))#读出延时
    spk_numpy1 = np.zeros((h, w, frame_end - frame_start + 10), dtype = np.uint8)
    temp_interval1 = np.ones((h, w))
    spike_interval1 = np.zeros((h, w, frame_end - frame_start + 10), dtype = int)
    f = open(out_filepath + "noise_pattern" + ".dat", 'wb') #输出有噪脉冲流文件
    if is_no_noise == True:
        f2 = open(out_filepath + "no_noise" + ".dat" , 'wb')#输出无噪脉冲流文件
        accumulator2 = np.random.rand(h, w)
        byte2 = 0
        pos2 = 0
        is_delay2 = np.zeros((h, w))
        spk_numpy2 = np.zeros((h, w, frame_end - frame_start + 10), dtype = np.uint8)
        temp_interval2 = np.ones((h, w))
        spike_interval2 = np.zeros((h, w, frame_end - frame_start + 10), dtype = int)
    temp_cnt = 0 #多个图像用于一帧 记录当前已经使用了几个图像
    is_spike_vis = 0 #是否使用脉冲可视化
    spike_vis = np.zeros((h, w))
    for i in range(frame_start, frame_end + 1):
        t_now = i
        temp_cnt += 1
        print(i)
        #num = str(i)
        num = str(i).zfill(4) #自动补0 总共补成到5位
        img_str = in_filepath + num + ".png"
        print(img_str)
        img = np.array(Image.open(img_str).convert('L').resize((w, h),Image.ANTIALIAS))
        print(img.shape)
        for a in range(h):#原本是for a in range h
            for b in range(w):#原本是for b in range w
                if img[a][b] <= 5e-4:
                    shot_noise = 0
                else:
                    shot_noise = np.random.poisson(lam= int(lux * img[a][b] / mx_dynamic_range), size=1) / (1.0 * int(lux * img[a][b] / mx_dynamic_range))
                #print("shot noise is ", shot_noise)
                temp_img = img[a][b] * dark * Alphas[a][b]
                #print(img[a][b])
                #print(temp_img)
                It[a][b] = noise_Idark[a][b] + shot_noise * 20 * CE * (Vrst - Vref + VE) * temp_img / mx_dynamic_range #实际输入电流 255对应一个脉冲
                #默认初始电压为(Vrst - Vref + VE) 然后噪声电压的期望是0
                temp_acc = 0.05 * It[a][b]
                #print("    ", It[a][b])
                if is_delay[a][b] != 0:
                    is_delay[a][b] = 0
                    temp_acc *= 0.998 #reset延时 一次读出时间是1/20000 一次reset是100ns 就是1e-7s
                accumulator[a][b] += temp_acc
                if is_no_noise == True:
                    temp_acc2 = 0.05 * (20 * CE * (Vrst - Vref + VE) * img[a][b] * dark / mx_dynamic_range) #用于生成无噪脉冲流
                    if is_delay2[a][b] != 0:
                        is_delay2[a][b] = 0
                        temp_acc2 *= 0.998 #reset延时 一次读出时间是1/20000 一次reset是100ns 就是1e-7s
                    accumulator2[a][b] += temp_acc2
                rand_vt = np.random.normal(0, Vt_hot_sigma) + np.random.normal(0, Vt_shot_sigma)
                #rand_vt = 0 #热噪声设置为0 用不用影响不大
                #Thres[a][b] = Cs[a][b] * (Vrst - Vref + Vs[a][b] + rand_vt)
                if accumulator[a][b] >= Cs[a][b] * (Vrst - Vref + Vs[a][b] + rand_vt):
                    spike_interval1[a,b,i - int(temp_interval1[a][b]) + 1 : i + 1] = int(temp_interval1[a][b])
                    accumulator[a][b] -= Cs[a][b] * (Vrst - Vref + Vs[a][b] + rand_vt)
                    is_delay[a][b] = 1
                    byte = byte | (1 << (pos))
                    spike_vis[a][b] = 255
                    spk_numpy1[a][b][t_now] = 1
                    temp_interval1[a][b] = 1
                else:
                    if i > frame_start:
                        spike_interval1[a][b][i] = spike_interval1[a][b][i - 1]
                    else:
                        spike_interval1[a][b][i] = 1000.0
                    temp_interval1[a][b] += 1
                    spike_vis[a][b] = 0
                    spk_numpy1[a][b][t_now] = 0
                pos += 1
                    #print(i)
                if pos == 8:
                    pos = 0
                    temp = struct.pack('B', byte)
                    #print(byte)
                    #print(struct.unpack('B', temp))
                    byte = 0
                    f.write(temp)
                #hidden1[a][b][0][t_now] = accumulator[a][b] / (CE * (Vrst - Vref + VE))
                if is_no_noise == True:
                    if accumulator2[a][b] >= CE * (Vrst - Vref + VE):
                        spike_interval2[a,b,i - int(temp_interval2[a][b]) + 1 : i + 1] = int(temp_interval2[a][b])
                        temp_interval2[a][b] = 1
                        accumulator2[a][b] -= CE * (Vrst - Vref + VE)
                        is_delay2[a][b] = 1
                        byte2 = byte2 | (1 << (pos2))
                        spk_numpy2[a][b][t_now] = 1
                    else:
                        if i > frame_start:
                            spike_interval2[a][b][i] = spike_interval2[a][b][i - 1]
                        else:
                            spike_interval2[a][b][i] = 1000
                        temp_interval2[a][b] += 1
                        spk_numpy2[a][b][t_now] = 0
                    pos2 += 1
                    if pos2 == 8:
                        pos2 = 0
                        temp2 = struct.pack('B', byte2)
                        #print(byte)
                        #print(struct.unpack('B', temp))
                        byte2 = 0
                        f2.write(temp2)
                    #hidden2[a][b][0][t_now] = accumulator2[a][b] / (CE * (Vrst - Vref + VE))
        if pos != 0:
            pos = 0
            temp = struct.pack('B', byte)
            byte = 0
            f.write(temp)
        if is_no_noise == True:
            if pos2 != 0:
                pos2 = 0
                temp2 = struct.pack('B', byte2)
                byte2 = 0
                f2.write(temp2)
    if is_no_noise == True:
        f2.close()
    #输出间隔
    for i in range(frame_start, frame_end + 1):
        np.save(out_filepath + "interval_noise_" + str(i) + ".npy", spike_interval1[:,:,i])
        if is_no_noise == True:
            np.save(out_filepath + "interval_gt_" + str(i) + ".npy", spike_interval2[:,:,i])
        index_left = i - int(wins / 2)
        index_right = i + int(wins / 2)
        if i >= int(wins / 2) and i <= frame_end - int(wins / 2):
            np.save(out_filepath + "noise_spike_" + str(i) + ".npy", spk_numpy1[:,:,index_left:index_right + 1])
            if is_no_noise == True:
                np.save(out_filepath + "no_spike_" + str(i) + ".npy", spk_numpy2[:,:,index_left:index_right + 1])
    return
if __name__=='__main__':
    t = np.uint8(10)
    print(t * -1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_start", "-fs", type = int, default=0)
    parser.add_argument("--frame_end", "-fe", type = int, default=1000)
    parser.add_argument("--class_start", "-cs", type = int, default=6)
    parser.add_argument("--class_end", "-ce", type = int, default=8)
    args = parser.parse_args()
    for i in range(args.class_start, args.class_end):
        dir_in =  ""#图像输入目录
        dir_out =  ""#脉冲输出目录
        if os.path.exists(dir_out) == False:
            os.mkdir(dir_out)
        get_spike(in_filepath = dir_in, out_filepath = dir_out, w = 400, h = 250, class_id = i, wins = 41, frame_start = 0, frame_end = 999)
