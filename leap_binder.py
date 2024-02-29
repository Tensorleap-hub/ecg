from typing import List, Tuple, Callable, Dict

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from code_loader import leap_binder
# Tensorleap imports
from code_loader.contract.datasetclasses import PreprocessResponse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from code_loader.contract.enums import LeapDataType
from ecg.utils.packages import install_all_packages
from ecg.utils.visualizers import *
import ast
# install_all_packages()
import wfdb

import tensorflow as tf
from ecg.data.preprocessing import *
from ecg.utils.gcs_utils import *
from ecg.utils.metrics import pred_label
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

LOCAL_STORAGE_PATH = local_file_path = os.path.join(os.getenv("HOME"), "Tensorleap_data", CONFIG['BUCKET_NAME'],
                                                    CONFIG['GCS_PATH'])

AGE_LABELS = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                        87, 88, 89, 90, 91, 92, 93, 94, 95]).astype(str).tolist()

indexes_to_delete = [0, 18, 20, 23, 26, 28, 34, 39, 56, 108, 119, 121, 122, 132, 138, 144, 181, 189, 200, 219, 227, 229, 238, 246, 255, 256, 281, 282, 298, 320, 326, 336, 346, 347, 378, 457, 458, 460, 510, 513, 519, 522, 525, 532, 535, 540, 569, 582, 598, 613, 634, 636, 637, 646, 688, 689, 699, 712, 727, 730, 757, 758, 761, 762, 763, 766, 798, 813, 814, 825, 828, 860, 887, 895, 907, 910, 923, 936, 948, 953, 973, 990, 997, 1020, 1046, 1057, 1063, 1068, 1075, 1077, 1099, 1107, 1111, 1115, 1164, 1173, 1185, 1189, 1231, 1233, 1237, 1240, 1241, 1252, 1255, 1261, 1268, 1270, 1273, 1279, 1291, 1292, 1300, 1356, 1420, 1473, 1494, 1513, 1516, 1541, 1551, 1566, 1569, 1597, 1616, 1622, 1625, 1647, 1654, 1669, 1671, 1681, 1689, 1694, 1701, 1704, 1707, 1712, 1721, 1726, 1747, 1748, 1795, 1844, 1858, 1877, 1916, 1921, 1941, 1942, 1947, 1964, 1965, 1971, 1972, 1998, 2008, 2019, 2041, 2050, 2051, 2054, 2059, 2064, 2073, 2092, 2096, 2103, 2116, 2139, 2142, 2180, 2193, 2211, 2240, 2243, 2251, 2252, 2264, 2293, 2322, 2349, 2354, 2360, 2401, 2411, 2428, 2430, 2447, 2448, 2476, 2485, 2486, 2500, 2507, 2513, 2556, 2577, 2592, 2596, 2610, 2614, 2629, 2635, 2669, 2681, 2692, 2724, 2739, 2741, 2769, 2784, 2786, 2805, 2823, 2833, 2848, 2860, 2862, 2864, 2868, 2869, 2881, 2889, 2894, 2897, 2900, 2906, 2914, 2925, 2929, 2934, 2954, 2990, 2997, 2999, 3000, 3025, 3029, 3051, 3062, 3069, 3088, 3097, 3117, 3135, 3148, 3150, 3161, 3181, 3185, 3198, 3204, 3209, 3238, 3248, 3267, 3292, 3295, 3303, 3306, 3321, 3338, 3341, 3352, 3368, 3371, 3390, 3428, 3481, 3484, 3489, 3496, 3499, 3503, 3509, 3534, 3542, 3551, 3553, 3557, 3558, 3562, 3566, 3569, 3575, 3586, 3589, 3590, 3597, 3598, 3608, 3615, 3628, 3629, 3634, 3642, 3654, 3694, 3704, 3728, 3734, 3737, 3771, 3776, 3792, 3795, 3796, 3799, 3802, 3816, 3817, 3827, 3833, 3864, 3869, 3897, 3908, 3913, 3914, 3924, 3926, 3939, 3942, 3968, 4014, 4056, 4090, 4092, 4095, 4097, 4099, 4103, 4106, 4110, 4112, 4119, 4147, 4158, 4179, 4181, 4196, 4212, 4218, 4227, 4230, 4233, 4240, 4251, 4255, 4261, 4276, 4283, 4311, 4316, 4322, 4360, 4363, 4365, 4376, 4379, 4380, 4391, 4423, 4431, 4449, 4463, 4465, 4467, 4489, 4492, 4495, 4499, 4502, 4506, 4525, 4531, 4532, 4533, 4538, 4547, 4560, 4575, 4578, 4583, 4590, 4595, 4611, 4615, 4637, 4643, 4647, 4648, 4656, 4666, 4701, 4703, 4729, 4732, 4737, 4745, 4750, 4772, 4773, 4791, 4792, 4793, 4797, 4805, 4809, 4815, 4827, 4829, 4830, 4838, 4840, 4859, 4875, 4878, 4902, 4907, 4908, 4916, 4918, 4922, 4940, 4957, 4972, 4974, 4977, 4983, 4984, 5003, 5012, 5017, 5040, 5045, 5047, 5048, 5058, 5059, 5060, 5061, 5065, 5071, 5080, 5088, 5096, 5098, 5104, 5131, 5143, 5147, 5152, 5154, 5159, 5166, 5173, 5176, 5180, 5193, 5196, 5198, 5201, 5204, 5208, 5210, 5217, 5218, 5219, 5220, 5221, 5224, 5238, 5242, 5245, 5249, 5252, 5257, 5260, 5261, 5262, 5263, 5268, 5270, 5274, 5278, 5279, 5289, 5293, 5310, 5314, 5321, 5322, 5330, 5340, 5341, 5342, 5345, 5350, 5352, 5354, 5356, 5360, 5365, 5372, 5374, 5378, 5390, 5407, 5408, 5413, 5417, 5421, 5424, 5431, 5433, 5435, 5446, 5450, 5451, 5455, 5461, 5469, 5471, 5473, 5481, 5482, 5485, 5494, 5517, 5521, 5522, 5525, 5526, 5528, 5529, 5531, 5536, 5547, 5548, 5549, 5550, 5556, 5566, 5572, 5573, 5583, 5590, 5599, 5603, 5607, 5612, 5613, 5614, 5616, 5620, 5628, 5635, 5636, 5637, 5639, 5643, 5646, 5647, 5648, 5655, 5658, 5665, 5666, 5667, 5670, 5679, 5680, 5691, 5693, 5694, 5697, 5700, 5702, 5710, 5712, 5718, 5724, 5727, 5743, 5758, 5766, 5769, 5772, 5785, 5787, 5790, 5800, 5804, 5807, 5816, 5858, 5871, 5873, 5879, 5890, 5894, 5895, 5906, 5925, 5940, 5958, 5966, 5974, 5997, 6017, 6023, 6027, 6042, 6064, 6091, 6115, 6120, 6122, 6126, 6134, 6137, 6142, 6151, 6172, 6184, 6201, 6205, 6208, 6210, 6218, 6221, 6236, 6238, 6255, 6257, 6268, 6270, 6297, 6303, 6313, 6320, 6325, 6331, 6343, 6353, 6360, 6396, 6404, 6405, 6412, 6415, 6424, 6447, 6458, 6467, 6468, 6484, 6520, 6528, 6531, 6545, 6548, 6558, 6560, 6561, 6575, 6577, 6627, 6651, 6666, 6681, 6687, 6701, 6702, 6728, 6734, 6767, 6794, 6809, 6812, 6815, 6816, 6817, 6819, 6820, 6821, 6823, 6825, 6828, 6833, 6841, 6844, 6856, 6866, 6867, 6868, 6872, 6875, 6877, 6892, 6983, 6985, 6993, 6998, 7000, 7016, 7027, 7050, 7052, 7055, 7057, 7058, 7067, 7077, 7079, 7094, 7098, 7102, 7103, 7107, 7118, 7124, 7133, 7138, 7148, 7155, 7158, 7159, 7180, 7183, 7186, 7188, 7194, 7214, 7219, 7228, 7229, 7231, 7232, 7233, 7244, 7246, 7249, 7252, 7254, 7259, 7261, 7265, 7268, 7272, 7273, 7274, 7276, 7280, 7281, 7295, 7298, 7302, 7304, 7307, 7310, 7311, 7314, 7325, 7328, 7333, 7336, 7339, 7340, 7342, 7345, 7350, 7356, 7359, 7360, 7361, 7365, 7366, 7379, 7387, 7389, 7393, 7399, 7403, 7404, 7407, 7411, 7412, 7422, 7423, 7424, 7428, 7431, 7433, 7436, 7438, 7439, 7441, 7445, 7450, 7454, 7457, 7462, 7463, 7465, 7467, 7469, 7471, 7474, 7480, 7495, 7496, 7497, 7501, 7506, 7508, 7512, 7517, 7520, 7522, 7525, 7526, 7527, 7529, 7530, 7542, 7544, 7545, 7567, 7578, 7579, 7582, 7586, 7591, 7592, 7594, 7595, 7600, 7613, 7616, 7623, 7625, 7626, 7630, 7631, 7633, 7634, 7638, 7645, 7648, 7653, 7654, 7657, 7673, 7675, 7676, 7685, 7688, 7700, 7706, 7707, 7714, 7729, 7730, 7731, 7734, 7736, 7741, 7742, 7746, 7751, 7755, 7777, 7778, 7780, 7783, 7811, 7813, 7824, 7828, 7835, 7881, 7884, 7885, 7890, 7893, 7909, 7914, 7961, 7962, 7991, 8007, 8018, 8029, 8030, 8032, 8072, 8092, 8104, 8105, 8119, 8145, 8146, 8156, 8160, 8166, 8184, 8202, 8209, 8212, 8213, 8219, 8222, 8233, 8246, 8250, 8252, 8265, 8267, 8297, 8326, 8357, 8360, 8376, 8382, 8386, 8413, 8420, 8421, 8430, 8431, 8469, 8475, 8477, 8480, 8493, 8501, 8502, 8512, 8513, 8521, 8533, 8557, 8575, 8581, 8590, 8595, 8596, 8597, 8620, 8626, 8636, 8641, 8652, 8657, 8668, 8670, 8672, 8675, 8678, 8690, 8691, 8695, 8698, 8709, 8731, 8752, 8753, 8760, 8763, 8769, 8806, 8808, 8809, 8812, 8820, 8828, 8835, 8840, 8848, 8850, 8851, 8859, 8883, 8892, 8897, 8914, 8920, 8925, 8927, 8937, 8989, 8990, 8993, 8999, 9003, 9012, 9021, 9029, 9034, 9036, 9040, 9044, 9046, 9049, 9051, 9059, 9072, 9075, 9086, 9089, 9102, 9107, 9129, 9138, 9141, 9144, 9148, 9151, 9152, 9153, 9154, 9161, 9166, 9167, 9168, 9174, 9196, 9205, 9207, 9209, 9212, 9217, 9219, 9220, 9222, 9227, 9232, 9253, 9265, 9268, 9269, 9279, 9289, 9294, 9295, 9297, 9298, 9312, 9315, 9316, 9322, 9326, 9350, 9357, 9360, 9365, 9366, 9375, 9380, 9395, 9397, 9400, 9405, 9408, 9409, 9412, 9418, 9420, 9421, 9424, 9435, 9436, 9437, 9441, 9442, 9443, 9444, 9445, 9446, 9451, 9459, 9460, 9474, 9485, 9510, 9511, 9512, 9518, 9519, 9532, 9539, 9545, 9548, 9550, 9556, 9561, 9564, 9578, 9583, 9585, 9586, 9588, 9589, 9591, 9592, 9595, 9596, 9598, 9602, 9615, 9620, 9627, 9629, 9634, 9641, 9643, 9650, 9654, 9655, 9656, 9657, 9662, 9665, 9682, 9696, 9697, 9698, 9701, 9706, 9710, 9715, 9732, 9742, 9745, 9746, 9753, 9754, 9757, 9764, 9769, 9784, 9822, 9828, 9857, 9866, 9881, 9888, 9893, 9900, 9904, 9908, 9924, 9928, 9932, 9975, 9976, 9981, 10001, 10011, 10022, 10054, 10074, 10078, 10109, 10111, 10121, 10166, 10177, 10179, 10201, 10223, 10225, 10241, 10242, 10247, 10254, 10263, 10293, 10300, 10301, 10302, 10304, 10305, 10306, 10342, 10346, 10349, 10362, 10390, 10398, 10408, 10410, 10446, 10464, 10531, 10548, 10553, 10554, 10563, 10565, 10566, 10570, 10628, 10639, 10657, 10659, 10662, 10670, 10706, 10707, 10712, 10721, 10724, 10728, 10730, 10734, 10766, 10774, 10775, 10785, 10799, 10833, 10857, 10875, 10878, 10888, 10893, 10895, 10897, 10902, 10913, 10933, 10935, 10941, 10943, 10944, 10958, 10964, 10965, 10986, 11001, 11006, 11023, 11024, 11025, 11046, 11101, 11104, 11119, 11150, 11160, 11164, 11165, 11183, 11185, 11192, 11199, 11205, 11206, 11217, 11218, 11223, 11239, 11251, 11252, 11269, 11272, 11279, 11285, 11288, 11302, 11330, 11333, 11337, 11339, 11340, 11342, 11344, 11351, 11355, 11357, 11375, 11380, 11386, 11388, 11389, 11393, 11396, 11397, 11398, 11405, 11409, 11411, 11413, 11420, 11424, 11427, 11429, 11430, 11435, 11442, 11443, 11447, 11450, 11451, 11458, 11461, 11462, 11466, 11473, 11475, 11479, 11481, 11496, 11497, 11504, 11508, 11512, 11522, 11536, 11539, 11541, 11546, 11550, 11559, 11566, 11568, 11569, 11570, 11571, 11573, 11589, 11590, 11591, 11592, 11602, 11606, 11607, 11612, 11619, 11620, 11628, 11632, 11633, 11634, 11635, 11639, 11641, 11645, 11649, 11650, 11652, 11653, 11655, 11656, 11658, 11659, 11668, 11670, 11676, 11677, 11680, 11683, 11690, 11699, 11706, 11714, 11715, 11717, 11719, 11720, 11728, 11730, 11732, 11740, 11751, 11761, 11782, 11785, 11787, 11805, 11815, 11816, 11821, 11831, 11839, 11843, 11854, 11864, 11869, 11874, 11875, 11887, 11893, 11898, 11899, 11901, 11914, 11915, 11922, 11934, 11937, 11944, 11948, 12013, 12028, 12032, 12039, 12065, 12081, 12085, 12108, 12123, 12156, 12173, 12195, 12197, 12203, 12208, 12214, 12225, 12228, 12231, 12242, 12244, 12248, 12254, 12259, 12265, 12281, 12300, 12308, 12332, 12335, 12356, 12385, 12386, 12391, 12395, 12406, 12416, 12417, 12421, 12426, 12442, 12446, 12447, 12454, 12459, 12484, 12494, 12504, 12506, 12507, 12509, 12519, 12520, 12523, 12525, 12531, 12532, 12545, 12555, 12561, 12572, 12596, 12604, 12609, 12620, 12632, 12639, 12641, 12643, 12646, 12649, 12671, 12673, 12676, 12690, 12710, 12730, 12736, 12737, 12750, 12758, 12774, 12785, 12793, 12797, 12800, 12808, 12813, 12821, 12822, 12827, 12830, 12831, 12856, 12865, 12866, 12870, 12871, 12909, 12922, 12923, 12931, 12935, 12940, 12948, 12958, 12971, 12974, 12978, 12987, 12993, 13001, 13012, 13021, 13033, 13037, 13046, 13058, 13060, 13069, 13076, 13087, 13097, 13104, 13107, 13111, 13112, 13116, 13130, 13131, 13134, 13137, 13159, 13166, 13170, 13173, 13174, 13177, 13184, 13185, 13187, 13189, 13196, 13200, 13218, 13219, 13222, 13235, 13252, 13259, 13261, 13263, 13266, 13275, 13279, 13280, 13288, 13290, 13301, 13307, 13308, 13309, 13321, 13324, 13327, 13328, 13330, 13334, 13337, 13339, 13340, 13350, 13359, 13360, 13364, 13367, 13372, 13375, 13381, 13382, 13391, 13394, 13395, 13396, 13398, 13418, 13437, 13453, 13455, 13458, 13460, 13468, 13477, 13492, 13495, 13500, 13504, 13506, 13508, 13509, 13515, 13526, 13527, 13531, 13535, 13540, 13553, 13558, 13559, 13561, 13562, 13565, 13567, 13568, 13569, 13573, 13576, 13577, 13579, 13581, 13584, 13586, 13588, 13598, 13599, 13609, 13610, 13617, 13629, 13635, 13646, 13653, 13655, 13667, 13671, 13728, 13732, 13743, 13749, 13753, 13755, 13761, 13779, 13782, 13789, 13792, 13793, 13794, 13798, 13800, 13814, 13820, 13825, 13828, 13834, 13853, 13855, 13858, 13862, 13873, 13879, 13885, 13893, 13895, 13914, 13918, 13939, 13943, 13955, 13962, 13967, 13987, 13988, 13994, 14009, 14020, 14045, 14053, 14054, 14055, 14096, 14115, 14156, 14159, 14168, 14171, 14172, 14174, 14184, 14189, 14196, 14203, 14221, 14228, 14261, 14285, 14296, 14311, 14315, 14323, 14327, 14331, 14341, 14354, 14361, 14364, 14370, 14381, 14383, 14392, 14400, 14421, 14422, 14424, 14437, 14449, 14472, 14475, 14480, 14482, 14497, 14535, 14542, 14544, 14547, 14555, 14561, 14568, 14569, 14602, 14603, 14625, 14639, 14646, 14655, 14660, 14661, 14692, 14696, 14699, 14710, 14716, 14736, 14741, 14747, 14763, 14764, 14765, 14766, 14768, 14774, 14780, 14789, 14792, 14794, 14798, 14800, 14802, 14822, 14824, 14825, 14843, 14844, 14847, 14851, 14857, 14860, 14862, 14868, 14887, 14890, 14906, 14913, 14954, 14963, 14970, 14979, 14986, 15021, 15023, 15026, 15029, 15040, 15042, 15047, 15064, 15065, 15080, 15098, 15135, 15139, 15145, 15152, 15160, 15168, 15179, 15180, 15182, 15183, 15194, 15195, 15196, 15209, 15213, 15215, 15216, 15236, 15237, 15240, 15241, 15243, 15244, 15259, 15261, 15262, 15270, 15279, 15302, 15308, 15314, 15315, 15318, 15324, 15333, 15334, 15335, 15337, 15344, 15352, 15357, 15360, 15361, 15362, 15364, 15366, 15369, 15385, 15386, 15391, 15392, 15399, 15404, 15405, 15417, 15419, 15423, 15427, 15431, 15434, 15437, 15438, 15442, 15452, 15454, 15469, 15470, 15471, 15472, 15474, 15476, 15491, 15492, 15504, 15508, 15513, 15514, 15516, 15517, 15519, 15520, 15528, 15532, 15537, 15544, 15549, 15551, 15553, 15559, 15563, 15567, 15568, 15570, 15571, 15573, 15574, 15583, 15584, 15591, 15592, 15594, 15597, 15599, 15601, 15602, 15604, 15608, 15611, 15612, 15615, 15618, 15619, 15621, 15625, 15626, 15627, 15628, 15635, 15636, 15637, 15639, 15642, 15645, 15646, 15647, 15659, 15666, 15672, 15681, 15692, 15699, 15703, 15710, 15723, 15751, 15760, 15785, 15786, 15805, 15811, 15815, 15817, 15822, 15825, 15829, 15843, 15845, 15846, 15848, 15849, 15857, 15866, 15878, 15879, 15900, 15909, 15921, 15937, 15945, 15959, 15960, 15966, 15970, 16003, 16007, 16021, 16026, 16031, 16042, 16046, 16103, 16107, 16108, 16110, 16123, 16135, 16140, 16159, 16168, 16187, 16191, 16213, 16244, 16256, 16264, 16282, 16315, 16322, 16324, 16325, 16338, 16347, 16364, 16365, 16366, 16367, 16372, 16402, 16404, 16405, 16408, 16409, 16420, 16423, 16428, 16433, 16449, 16450, 16451, 16457, 16462, 16468, 16478, 16479, 16480, 16481, 16488, 16490, 16493, 16495, 16503, 16506, 16530, 16552, 16565, 16570, 16595, 16604, 16620, 16636, 16638, 16639, 16645, 16657, 16658, 16663, 16668, 16670, 16672, 16680, 16693, 16706, 16708, 16713, 16714, 16715, 16719, 16721, 16724, 16725, 16728, 16733, 16748, 16757, 16776, 16785, 16805, 16820, 16834, 16840, 16843, 16858, 16862, 16867, 16875, 16880, 16888, 16896, 16905, 16916, 16924, 16926, 16927, 16933, 16936, 16942, 16948, 16949, 16951, 16957, 16969, 16975, 16979, 16982, 16985, 16986, 16990, 17001, 17002, 17007, 17012, 17017, 17032, 17035, 17040, 17042, 17053, 17057, 17058, 17060, 17067, 17068, 17079, 17080, 17084, 17091, 17092, 17094, 17095, 17096, 17098, 17102, 17103, 17106, 17107, 17111, 17112, 17113, 17117, 17119, 17122, 17124, 17125, 17132, 17136, 17137, 17138, 17141, 17142, 17143, 17144, 17145, 17155, 17158, 17160, 17164, 17172, 17173, 17175, 17178, 17186, 17192, 17195, 17196, 17199, 17206, 17239, 17241, 17250, 17251, 17254, 17256, 17272, 17280, 17288, 17295, 17298, 17302, 17310, 17312, 17319, 17331, 17334, 17336, 17338, 17344, 17352, 17353, 17355, 17356, 17358, 17360, 17362, 17364, 17366, 17367, 17372, 17376, 17378, 17379, 17381, 17382, 17383, 17384, 17386, 17387, 17390, 17397, 17403, 17406, 17407, 17409, 17413, 17416, 17418, 17425, 17426, 17427, 17428, 17429, 17430, 17431, 17433, 17437, 17445, 17447, 17449, 17451, 17456, 17461, 17464, 17466, 17477, 17482, 17496, 17497, 17500, 17504, 17507, 17508, 17518, 17535, 17539, 17558, 17561, 17566, 17568, 17581, 17583, 17598, 17601, 17613, 17623, 17626, 17633, 17634, 17653, 17660, 17673, 17680, 17681, 17682, 17707, 17715, 17749, 17769, 17789, 17816, 17868, 17889, 17893, 17912, 17920, 17924, 17928, 17932, 17934, 17937, 17943, 17944, 17952, 17956, 17957, 17985, 17989, 18017, 18047, 18050, 18054, 18057, 18075, 18083, 18084, 18085, 18086, 18093, 18105, 18110, 18140, 18151, 18155, 18156, 18163, 18164, 18166, 18178, 18194, 18204, 18217, 18228, 18246, 18248, 18266, 18280, 18282, 18293, 18298, 18303, 18309, 18311, 18325, 18328, 18339, 18347, 18348, 18379, 18385, 18389, 18401, 18421, 18429, 18430, 18434, 18435, 18448, 18458, 18464, 18470, 18471, 18500, 18508, 18512, 18526, 18528, 18530, 18533, 18548, 18557, 18570, 18574, 18580, 18581, 18590, 18601, 18604, 18606, 18608, 18618, 18642, 18647, 18660, 18675, 18677, 18678, 18679, 18687, 18691, 18698, 18701, 18713, 18715, 18716, 18717, 18722, 18726, 18729, 18730, 18742, 18752, 18755, 18756, 18758, 18760, 18771, 18772, 18775, 18777, 18782, 18783, 18794]


# Preprocess Function:
def preprocess_func() -> Tuple[PreprocessResponse, PreprocessResponse]:
    path = CONFIG['GCS_PATH']
    csv_path = _download(f'{path}/ptbxl_database.csv', f'{LOCAL_STORAGE_PATH}/ptbxl_database.csv')
    # csv_path = '/Users/chenrothschild/Tensorleap/data/example-datasets-47ml982d/ecg/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database_new.csv'
    # load and convert annotation data
    data = pd.read_csv(csv_path)
    # data.scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))

    data = data.drop_duplicates(subset=['patient_id']).reset_index(drop=True)
    data = data.dropna(subset=['age', 'sex']).reset_index(drop=True)
    data = data.drop(indexes_to_delete).reset_index(drop=True)
    data.fillna(-1, inplace=True)
    data = add_diagnostic(data)

    train = data[data['strat_fold'] <= 8]
    val = data[data['strat_fold'] == 9]
    # test = data[data['strat_fold'] == 10]

    train = train.iloc[:CONFIG['TRAIN_SAMPLE_SIZE']] if CONFIG['TRAIN_SAMPLE_SIZE'] is not None else train
    val = val.iloc[:CONFIG['VAL_SAMPLE_SIZE']] if CONFIG['VAL_SAMPLE_SIZE'] is not None else val
    # test = test.iloc[:CONFIG['TEST_SAMPLE_SIZE']] if CONFIG['TEST_SAMPLE_SIZE'] is not None else test

    train = PreprocessResponse(length=len(train) * CONFIG['SAMPLE_TIMES'], data={'images': train, 'set': 'train'})
    val = PreprocessResponse(length=len(val) * CONFIG['SAMPLE_TIMES'], data={'images': val, 'set': 'val'})
    # test = PreprocessResponse(length=len(test) * CONFIG['SAMPLE_TIMES'], data=test)

    return train, val


def unlabeled_data() -> PreprocessResponse:
    path = CONFIG['GCS_PATH']
    csv_path = _download(f'{path}/ptbxl_database.csv', f'{LOCAL_STORAGE_PATH}/ptbxl_database.csv')
    # csv_path = '/Users/chenrothschild/Tensorleap/data/example-datasets-47ml982d/ecg/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database_new.csv'
    # load and convert annotation data
    data = pd.read_csv(csv_path)
    # data.scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))

    data = data.drop_duplicates(subset=['patient_id']).reset_index(drop=True)
    data = data.dropna(subset=['age', 'sex']).reset_index(drop=True)
    data = data.drop(indexes_to_delete).reset_index(drop=True)
    data.fillna(-1, inplace=True)
    data = add_diagnostic(data)

    unlabeled = data[data['strat_fold'] == 10]

    unlabeled = unlabeled.iloc[:CONFIG['TEST_SAMPLE_SIZE']] if CONFIG['TEST_SAMPLE_SIZE'] is not None else unlabeled

    unlabeled = PreprocessResponse(length=len(unlabeled) * CONFIG['SAMPLE_TIMES'], data={'images': unlabeled, 'set': 'unlabeled'})

    return unlabeled




def add_diagnostic(df):
    # Load scp_statements.csv for diagnostic aggregation
    path = CONFIG['GCS_PATH']
    csv_path = _download(f'{path}/scp_statements.csv', f'{LOCAL_STORAGE_PATH}/scp_statements.csv')
    # csv_path = '/Users/chenrothschild/Tensorleap/data/example-datasets-47ml982d/ecg/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv'

    agg_df = pd.read_csv(csv_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y):
        tmp = []
        y_dic = ast.literal_eval(y)
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        if len(tmp) == 0:
            tmp = ['None']
        return list(set(tmp))

    # Apply diagnostic superclass
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(lambda x: x[0])
    df = pd.get_dummies(df, prefix=['diagnostic_superclass'], columns=['diagnostic_superclass'])
    return df


def get_sample_ind(idx: int) -> int:
    return int(idx // CONFIG['SAMPLE_TIMES'])


# Input encoder fetches the time-series with the index `idx` from the data from set in
# the PreprocessResponse's data. Returns a ndarray containing the sample's tokens.
def input_encoder_raw(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample_ind = idx

    filename_hr = preprocess.data['images'].iloc[sample_ind]['filename_hr']
    path = CONFIG['GCS_PATH']
    prefix = f'{path}/{filename_hr}'
    csv_path = _download(f'{prefix}.hea', f'{LOCAL_STORAGE_PATH}/{filename_hr}.hea')
    csv_path = _download(f'{prefix}.dat', f'{LOCAL_STORAGE_PATH}/{filename_hr}.dat')

    data = wfdb.rdsamp(f'{LOCAL_STORAGE_PATH}/{filename_hr}')
    # data = wfdb.rdsamp(
    #     f"/Users/chenrothschild/Tensorleap/data/example-datasets-47ml982d/ecg/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/{filename_hr}")

    data = np.array(data[0])
    # scaler = MinMaxScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)
    # data = np.transpose(data)  # if cnn2heads

    data = data[..., np.newaxis]
    data = tf.image.per_image_standardization(data)

    return np.squeeze(data.numpy(), axis=2)
    # return data.numpy()


# Input encoder fetches the time-series with the index `idx` from the data from set in
# the PreprocessResponse's data. Returns an ndarray containing the sample's tokens.
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample_ind = get_sample_ind(idx)
    filename_hr = preprocess.data.iloc[sample_ind]['filename_hr']
    path = CONFIG['GCS_PATH']
    prefix = f'{path}/{filename_hr}'
    csv_path = _download(f'{prefix}.hea', f'{LOCAL_STORAGE_PATH}/{filename_hr}.hea')
    csv_path = _download(f'{prefix}.dat', f'{LOCAL_STORAGE_PATH}/{filename_hr}.dat')

    data = wfdb.rdsamp(f'{LOCAL_STORAGE_PATH}/{filename_hr}')
    # data = wfdb.rdsamp(
    #     f"/Users/chenrothschild/Tensorleap/data/example-datasets-47ml982d/ecg/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/{filename_hr}")

    data = np.array(data[0])
    # scaler = MinMaxScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)
    data = data[np.newaxis, ..., np.newaxis]
    patch_h, patch_w = 224, 12
    patches = tf.image.extract_patches(
        images=data,
        sizes=[1, patch_h, patch_w, 1],
        strides=[1, patch_h, patch_w, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )

    # per sample, we split into 23 time windows of 224x12
    # shape is [batch, sample_times, 1, patch_flatten]
    slice_ind = int(idx % CONFIG['SAMPLE_TIMES'])
    sliced = patches[:, slice_ind, :, :]
    sliced = tf.reshape(sliced, [1, patch_h, patch_w])
    sliced = tf.transpose(sliced, perm=[1, 2, 0])
    # slice = tf.image.resize(slice, [224, 224])
    sliced = tf.image.per_image_standardization(sliced)

    return sliced.numpy()


# Ground truth encoder fetches the label with the index `idx` from the `toxicity` column set in
# the PreprocessResponse's data. Returns a hot vector class
def gt_scp(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    # if preprocess.data['set'] == 'unlabeled':
    #     return np.array([0, 0, 0, 0, 0, 0])
    # else:
    idx = get_sample_ind(idx)
    gt = preprocess.data['images'].iloc[idx].filter(regex='diagnostic_superclass')
    gt = gt.to_numpy()
    return gt.astype(int)


def gt_age(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].age


def gt_gender(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    idx = get_sample_ind(idx)
    return np.eye(2)[int(preprocess.data['images'].iloc[idx].sex)]


# ----------------------------------metadata--------------------------------------
def gender_metadata(idx: int, preprocess: PreprocessResponse) -> str:
    idx = get_sample_ind(idx)
    if preprocess.data['images'].iloc[idx].sex:
        return 'male'
    else:
        return 'female'


def gender_metadata_int(idx: int, preprocess: PreprocessResponse) -> int:
    idx = get_sample_ind(idx)
    if preprocess.data['images'].iloc[idx].sex:
        return 1
    else:
        return 0


def age_metadata(idx: int, preprocess: PreprocessResponse) -> float:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].age


def height_metadata(idx: int, preprocess: PreprocessResponse) -> float:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].height


def weight_metadata(idx: int, preprocess: PreprocessResponse) -> float:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].weight


def device_metadata(idx: int, preprocess: PreprocessResponse) -> str:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].device


def nurse_metadata(idx: int, preprocess: PreprocessResponse) -> float:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].nurse


def site_metadata(idx: int, preprocess: PreprocessResponse) -> str:
    idx = get_sample_ind(idx)
    return preprocess.data['images'].iloc[idx].site


def validated_by_human_metadata(idx: int, preprocess: PreprocessResponse) -> int:
    idx = get_sample_ind(idx)
    return int(preprocess.data['images'].iloc[idx].validated_by_human)


def scp_code_metadata_dict(idx: int, preprocess: PreprocessResponse) -> Dict[str, int]:
    scp_codes_superclass = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    idx = get_sample_ind(idx)
    res = {}
    for code in scp_codes_superclass:
        if preprocess.data['set'] == 'unlabeled':
            res[f'scp_code_{code}'] = int(0)
        else:
            res[f'scp_code_{code}'] = int(preprocess.data['images'].iloc[idx][f'diagnostic_superclass_{code}'])
    return res


def get_scp_gt(idx: int, preprocess: PreprocessResponse) -> str:
    if preprocess.data['set'] == 'unlabeled':
        return 'unlabeled'
    else:
        idx = get_sample_ind(idx)
        gt = preprocess.data['images'].iloc[idx].filter(regex='diagnostic_superclass')
        column_name = [gt.index[i] for i, value in enumerate(gt) if value]
        return column_name[0].split('_')[-1]


def detect_p_complexes(ecg_signal, r_peaks):
    p_complexes = []
    for r_peak in r_peaks:
        # Search for the preceding P complex within a specified time window before the R peak
        p_start_index = max(0, r_peak - 70)  # Adjust the window size as needed
        p_end_index = r_peak
        p_complex_amplitude = np.max(ecg_signal[p_start_index:p_end_index])
        p_complexes.append((p_start_index, p_end_index, p_complex_amplitude))
    return p_complexes


def detect_t_complexes(ecg_signal, r_peaks):
    t_complexes = []
    for r_peak in r_peaks[:-1]:
        # Search for the T complex within a specified time window after the R peak
        t_start_index = min(len(ecg_signal), r_peak + 135)  # Example: Adjust the window size as needed
        t_end_index = min(len(ecg_signal), r_peak + 250)  # Example: Adjust the window size as needed
        t_complex_amplitude = np.max(ecg_signal[t_start_index:t_end_index])
        t_complexes.append((t_start_index, t_end_index, t_complex_amplitude))
    return t_complexes


def detect_qrs_complexes(ecg_signal, r_peaks):
    qrs_complexes = []
    for r_peak in r_peaks:
        # Search for the QRS complex within a specified time window around the R peak
        q_start_index = max(0, r_peak - 10)  # Example: Adjust the window size as needed
        q_end_index = min(len(ecg_signal), r_peak + 10)  # Example: Adjust the window size as needed
        qrs_complex_amplitude = np.max(ecg_signal[q_start_index:q_end_index])
        qrs_complexes.append((q_start_index, q_end_index, qrs_complex_amplitude))
    return qrs_complexes


def waveform_metadata_dict(idx: int, preprocess: PreprocessResponse, arg: str) -> Dict[str, float]:
    idx = get_sample_ind(idx)
    res = {}

    for i, signal_name in enumerate(CONFIG['sig_names']):
        ecg_signal = input_encoder_raw(idx, preprocess)[:, i]
        ecg_signal = ecg_signal[..., np.newaxis]

        r_peaks_x_axis, _ = find_peaks(ecg_signal[:, 0], distance=500)
        if r_peaks_x_axis.any():
            r_peaks_y_axis = ecg_signal[r_peaks_x_axis]

            p_peaks_x_axis = detect_p_complexes(ecg_signal, r_peaks_x_axis)
            p_peak_indices = [peak[0] for peak in p_peaks_x_axis]
            p_peaks_y_axis = ecg_signal[p_peak_indices]

            t_peaks_x_axis = detect_t_complexes(ecg_signal, r_peaks_x_axis)
            t_peak_indices = [peak[0] for peak in t_peaks_x_axis]
            t_peaks_y_axis = ecg_signal[t_peak_indices]

            qrs_complexes_x_axis = detect_qrs_complexes(ecg_signal, r_peaks_x_axis)
            qrs_complexes_indices = [complexes[:-1] for complexes in qrs_complexes_x_axis]
            a = [x-y for x,y in qrs_complexes_indices]

            if arg == 'mean':
                res[f'r_peaks_{signal_name}'] = np.round(np.mean(r_peaks_y_axis), 2).astype(float)
                res[f'p_peaks_{signal_name}'] = np.round(np.mean(p_peaks_y_axis), 2).astype(float)
                res[f't_peaks_{signal_name}'] = np.round(np.mean(t_peaks_y_axis), 2).astype(float)
            elif arg == 'min':
                res[f'r_peaks_{signal_name}'] = (np.min(r_peaks_y_axis)).astype(float)
                res[f'p_peaks_{signal_name}'] = (np.min(p_peaks_y_axis)).astype(float)
                res[f't_peaks_{signal_name}'] = (np.min(t_peaks_y_axis)).astype(float)
            else:
                res[f'r_peaks_{signal_name}'] = (np.max(r_peaks_y_axis)).astype(float)
                res[f'p_peaks_{signal_name}'] = (np.max(p_peaks_y_axis)).astype(float)
                res[f't_peaks_{signal_name}'] = (np.max(t_peaks_y_axis)).astype(float)
        else:
            res[f'{signal_name}'] = float(-100)

        # plt.figure(figsize=(12, 6))
        # plt.plot(ecg_signal, color='blue', label='ECG signal')
        #
        # # Plot R peaks on the signal
        # plt.scatter(r_peaks_x_axis, r_peaks_y_axis, color='red', label='R peaks', marker='o')
        # # Plot P peaks on the signal
        # plt.scatter(p_peak_indices, p_peaks_y_axis, color='green', label='P peaks', marker='o')
        # # Plot T peaks on the signal
        # plt.scatter(t_peak_indices, t_peaks_y_axis, color='black', label='T peaks', marker='o')
        # # Plot QRS peaks on the signal
        # for start, end in qrs_complexes_indices:
        #     plt.axvspan(start, end, color='purple', alpha=0.3)
        #
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.title('ECG Signal with R, T, P and QRS Peaks')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    return res


def waveform_mean_metadata_dict(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    idx = get_sample_ind(idx)
    res = waveform_metadata_dict(idx, preprocess, 'mean')
    return res


def waveform_min_metadata_dict(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    idx = get_sample_ind(idx)
    res = waveform_metadata_dict(idx, preprocess, 'min')
    return res


def waveform_max_metadata_dict(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    idx = get_sample_ind(idx)
    res = waveform_metadata_dict(idx, preprocess, 'max')
    return res


# Dataset binding functions to bind the functions above to the `Dataset`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_unlabeled_data_preprocess(function=unlabeled_data)

leap_binder.set_input(function=input_encoder_raw, name='ecg')
leap_binder.set_ground_truth(function=gt_scp, name='scp')
leap_binder.add_prediction(name='scp_label', labels=CONFIG['LABELS'])

leap_binder.set_metadata(function=gender_metadata, name='gender')
leap_binder.set_metadata(function=age_metadata, name='age')
leap_binder.set_metadata(function=height_metadata, name='height')
leap_binder.set_metadata(function=weight_metadata, name='weight')
leap_binder.set_metadata(function=device_metadata, name='device')
leap_binder.set_metadata(function=nurse_metadata, name='nurse')
leap_binder.set_metadata(function=site_metadata, name='site')
leap_binder.set_metadata(function=validated_by_human_metadata, name='validated_by_human')
leap_binder.set_metadata(function=gender_metadata_int, name='gender_int')
leap_binder.set_metadata(function=scp_code_metadata_dict, name='scp_code_dict')
leap_binder.set_metadata(function=waveform_mean_metadata_dict, name='waveform_mean')
leap_binder.set_metadata(function=waveform_min_metadata_dict, name='waveform_min')
leap_binder.set_metadata(function=waveform_max_metadata_dict, name='waveform_max')
leap_binder.set_metadata(function=get_scp_gt, name='scp_gt')

leap_binder.add_custom_metric(function=pred_label, name='pred_label')

for sig_name in CONFIG['sig_names']:
    visualizer_name = f'waveform_{sig_name}'
    visualizer_function = globals()[f'display_waveform_{sig_name}']
    heatmap_visualizer_function = globals()[f'display_waveform_{sig_name}_heatmap']

    leap_binder.set_visualizer(visualizer_function, name=visualizer_name,
                               heatmap_visualizer=heatmap_visualizer_function,
                               visualizer_type=LeapDataType.Graph)

leap_binder.set_visualizer(horizontal_bar_visualizer_with_labels_name, 'horizontal_bar_lm', LeapDataType.HorizontalBar)
leap_binder.set_visualizer(img_vis_T, 'img_visualizer', LeapDataType.Image, heatmap_visualizer=heatmap_vis_T)
leap_binder.set_visualizer(graph_visuaizer, 'graph_visualizer', LeapDataType.Graph,
                           heatmap_visualizer=graph_heatmap_visualizer)

if __name__ == "__main__":
    leap_binder.check()
