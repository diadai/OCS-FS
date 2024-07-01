import numpy as np

from load_data import ReadData as rd
from process_data import ProcessData
import cross_validation as cr
from compare import Comparison
import time
from src.reduce import Select
import open_learning
from collections import Counter


import sys
import os
np.random.seed(0)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def ocs(path, name):

    # os_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前py文件的父目录
    # type = sys.getfilesystemencoding()
    # sys.stdout = Logger(r'../result/'+name+'222.txt')
    # bank.csv, shuttle.csv, Sensorless.csv, connect-4.csv, har-PUC-Rio-ugulino.csv, miniboone_pid.csv
    path = r'../data\\' + name + '.csv'
    # Read data
    all_data, all_label, data_with_label = rd(path, 1).read_data()
    class_dict = Counter(all_label[:, 0])
    class_list = []
    for item in class_dict:
        class_list.append(item)
    class_k = len(class_list)

    k_folds = 10
    temp_train_index, temp_test_index = cr.cross_validation(all_data.shape[0], k_folds)

    init_data_ratio = [0.6] #0.3,
    new_data_ratio = [0.4] #, 0.3, 0.1

    result1 = []

    for g in range(1):
        for l in range(1):
            for f in range(1):  # change the Gaussian kernel parameter
                # if f < 10:
                #     para_s = (f + 1) * 0.1
                # else:
                #     para_s = f - 8
                para_s = 2
                print("para_s", para_s)
                para_init_data_ratio = init_data_ratio[g]
                para_new_data_ratio = new_data_ratio[l]
                print("para_s:", para_s)
                print("set the data ratio:", para_init_data_ratio, para_new_data_ratio)

                # Simulate open world data, 0.6 is the ratio of initial data, 0.3 is the ratio of new class
                pro_data = ProcessData(all_data, all_label, data_with_label, para_init_data_ratio, para_new_data_ratio)
                init_data, init_label, init_class_label = pro_data.initial_class()
                new_data_list, new_label_list, new_class_label_list = pro_data.new_class()

                # all data feature selection
                ori_select = Select(all_data, all_label, para_s, class_k, class_list, para_k=3)
                para_radius = 0.025
                while para_radius < 0.2:
                    if para_radius == 0.025:
                        start_time1 = time.time()
                        # Select = Select(all_data, all_label, s, class_k, class_list, para_k=3)
                        ori_find_sv_data, _ = ori_select.find_all_data_sv()
                    # all data feature selection
                    original_select, original_remian = ori_select.get_attribute_importance_theta(para_radius, ori_find_sv_data)
                    print("original_select,original_remain", original_select, original_remian)

                    # initial knowledge base: build granular balls and select features on initial data
                    open_select = Select(init_data, init_label, para_s, len(init_class_label), list(init_class_label), para_k=3)
                    open_find_sv_data, init_hypersphere = open_select.find_all_data_sv()
                    open_select, open_remian = open_select.get_attribute_importance_theta(para_radius, open_find_sv_data)
                    print("init select_att, remain_att:", open_select, open_remian)

                    # Open learning for new data sequences
                    for i in range(len(new_data_list)):
                        # Class recognition on the open-set
                        class_ident = open_learning.ClassIdentification(init_hypersphere, new_data_list[i], para_s,
                                                                        len(init_class_label), init_class_label)
                        unknown_data, known_data = class_ident.class_identification(list(range(all_data.shape[1])))

                        # Clustering unknown data and build new granular balls
                        new_hypersphere, cluster_data, cluster_label, cluster_label_list, open_sv = open_learning.BallUpdate(
                            unknown_data).new_balls("dbscan", para_s, i)
                        ball_update = open_learning.BallUpdate(unknown_data).knowledge_update(init_hypersphere, new_hypersphere)
                        sampling_update = open_learning.BallUpdate(unknown_data).sampling_update(open_find_sv_data, open_sv)

                        # Update feature selection subspace
                        if len(open_remian) == 0:
                            break
                        else:
                            open_select, open_remian = open_learning.SelectFeature(open_select, open_remian).select_feature2(para_radius, sampling_update)
                            print("new turn select_att, remain_att:", open_select, open_remian)

                    para_radius += 0.1
                    temp_accuracy_comparison = []
                    temp_f1_score = []
                    temp_cost_time = []

                    temp_accuracy_selected_comparison = []
                    temp_f1_selected_score = []
                    temp_selected_cost_time = []

                    HNRS_accuracy_selected_comparison = []
                    HNRS_f1_selected_score = []
                    HNRS_selected_cost_time = []

                    if len(open_select) == len(all_data[0]):
                        continue
                    for j in range(10):
                        # print(j)
                        train_data = all_data[temp_train_index[j]]
                        train_label = all_label[temp_train_index[j]]

                        test_data = all_data[temp_test_index[j]]
                        test_label = all_label[temp_test_index[j]]
                        # initial data
                        temp_accuracy_list_original, temp_f1_score_list_original, temp_cost_time_original = Comparison(
                            train_data,
                            train_label,
                            test_data,
                            test_label).comparison()
                        temp_accuracy_comparison.append(temp_accuracy_list_original)
                        temp_f1_score.append(temp_f1_score_list_original)
                        temp_cost_time.append(temp_cost_time_original)
                        # selected data

                        HNRS_accuracy_list_select, HNRS_f1_score_list_select, HNRS_cost_time_select = Comparison(
                            train_data[:, original_select],
                            train_label, test_data[:, original_select], test_label).comparison()

                        temp_accuracy_list_select, temp_f1_score_list_select, temp_cost_time_select = Comparison(
                            train_data[:, open_select],
                            train_label, test_data[:, open_select], test_label).comparison()

                        temp_accuracy_selected_comparison.append(temp_accuracy_list_select)
                        temp_f1_selected_score.append(temp_f1_score_list_select)
                        temp_selected_cost_time.append(temp_cost_time_select)

                        HNRS_accuracy_selected_comparison.append(HNRS_accuracy_list_select)
                        HNRS_f1_selected_score.append(HNRS_f1_score_list_select)
                        HNRS_selected_cost_time.append(HNRS_cost_time_select)

                    for k in range(5):
                        final_score_comparison = []
                        final_f1_score = []
                        final_cost_time = []

                        final_score_selected_comparison = []
                        final_f1_score_selected = []
                        final_cost_time_selected = []

                        HNRS_score_selected_comparison = []
                        HNRS_f1_score_selected = []
                        HNRS_cost_time_selected = []

                        for z in range(10):
                            final_score_comparison.append(temp_accuracy_comparison[z][k])
                            final_f1_score.append(temp_f1_score[z][k])
                            final_cost_time.append(temp_cost_time[z][k])

                            final_score_selected_comparison.append(temp_accuracy_selected_comparison[z][k])
                            final_f1_score_selected.append(temp_f1_selected_score[z][k])
                            final_cost_time_selected.append(temp_selected_cost_time[z][k])

                            HNRS_score_selected_comparison.append(HNRS_accuracy_selected_comparison[z][k])
                            HNRS_f1_score_selected.append(HNRS_f1_selected_score[z][k])
                            HNRS_cost_time_selected.append(HNRS_selected_cost_time[z][k])

                        score_test_comparison = sum(final_score_comparison) / 10
                        temp_standard_comparison = np.std(final_score_comparison)

                        # print("ever_acc::", final_score_selected_comparison)

                        f1 = sum(final_f1_score) / 10
                        temp_f1_standard = np.std(final_f1_score)
                        average_time = sum(final_cost_time) / 10
                        print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (
                            score_test_comparison,
                            temp_standard_comparison,
                            f1,
                            temp_f1_standard,
                            average_time))
                        score_HNRS_selected_comparison = sum(HNRS_score_selected_comparison) / 10
                        HNRS_standard_selected_comparison = np.std(HNRS_score_selected_comparison)
                        HNRS_f1_selected = sum(HNRS_f1_score_selected) / 10
                        HNRS_f1_sel_standard = np.std(HNRS_f1_score_selected)
                        HNRS_average_time_selected = sum(HNRS_cost_time_selected) / 10
                        print("hnr acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (
                            score_HNRS_selected_comparison,
                            HNRS_standard_selected_comparison,
                            HNRS_f1_selected,
                            HNRS_f1_sel_standard,
                            HNRS_average_time_selected))

                        score_test_selected_comparison = sum(final_score_selected_comparison) / 10
                        temp_standard_selected_comparison = np.std(final_score_selected_comparison)
                        f1_selected = sum(final_f1_score_selected) / 10
                        temp_f1_sel_standard = np.std(final_f1_score_selected)
                        average_time_selected = sum(final_cost_time_selected) / 10
                        print("cfs acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (
                            score_test_selected_comparison,
                            temp_standard_selected_comparison,
                            f1_selected,
                            temp_f1_sel_standard,
                            average_time_selected))

                        result0 = {
                            "s": para_s,
                            "init_r": init_data_ratio[g],
                            "new_r": new_data_ratio[l],
                            "k_classify": k,
                            "ever_acc": final_score_selected_comparison,
                            "o_a": score_test_comparison,
                            "o_s": temp_standard_comparison,
                            "o_f1": f1,
                            "o_fs": temp_f1_standard,
                            "h_a": score_HNRS_selected_comparison,
                            "h_s": HNRS_standard_selected_comparison,
                            "h_f1": HNRS_f1_selected,
                            "h_fs": HNRS_f1_sel_standard,
                            "cfs_a": score_test_selected_comparison,
                            "cfs_s": temp_standard_selected_comparison,
                            "cfs_f1": f1_selected,
                            "cfs_fs": temp_f1_sel_standard,

                        }
                        result1.append(result0)
                        print("-----")
                    print("!!!!!!!new radius!!!!!!!!!!")
                print("--------------------each gaussian kernel parameter---------------")
            print("--------------------each init_data_ratio---------------")
        print("***************************each new data ratio********************")
    print("--------------------------------------------------------------------------------")
    return result1


