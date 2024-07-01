from src.reduce import Select
from sklearn import cluster
import numpy as np
from scipy.spatial import distance
import src.knsi0516 as knsi0516
from collections import Counter
from sklearn.neighbors import NearestNeighbors


class ClassIdentification:
    """
    identify the new data
    """
    def __init__(self, para_hypersphere, data, para_s, class_num, class_list):
        self.hypersphere = para_hypersphere
        self.data = data
        self.s = para_s
        self.class_num = class_num
        self.class_list = class_list

    def get_matrix(self, para_a, para_b):
        """
        calculate the Gaussian kernel distance between two vectors
        """
        dist_sq = distance.cdist(para_a, para_b)
        cur_sim_vec = np.exp(-np.square(dist_sq) / (2.0 * self.s * self.s))
        return cur_sim_vec

    def get_distance(self, att):
        """
        calculate the distance between the new data and the center of each hypersphere
        return:
        """
        distance_list = []
        for i in range(self.class_num):
            a = self.get_matrix(self.data[:, att], self.hypersphere["sv"][i][:, att])
            term_2 = -2 * np.dot(self.get_matrix(self.data[:, att], self.hypersphere["sv"][i][:, att]), self.hypersphere["alpha"][i])
            cur_sim_vec = self.get_matrix(self.hypersphere["sv"][i][:, att], self.hypersphere["sv"][i][:, att])
            term_3 = np.dot(np.dot(self.hypersphere["alpha"][i].T, cur_sim_vec), self.hypersphere["alpha"][i])
            a = 1 + term_2 + term_3
            temp_distance = np.sqrt(1 + term_2 + term_3)
            distance_list.append(temp_distance)
        return distance_list

    def class_identification(self, att):
        """

        """
        dis_list = self.get_distance(att)
        data_index = list(range(self.data.shape[0]))
        unknown_data_index = data_index
        known_data = []
        for i in range(self.class_num):
            compare_distance = np.where(dis_list[i] <= self.hypersphere["radius"][i])[0]  # 超球是否会存在空间重叠情况？
            # find the unknown data
            if compare_distance.all() not in unknown_data_index:
                continue
            if len(compare_distance) == 0:
                unknown_data_index = unknown_data_index
            else:
                known_data_label = np.tile(np.array(self.class_list[i]), len(compare_distance)).reshape((len(compare_distance)), 1)
                known_data = np.hstack((self.data[compare_distance, :], known_data_label))
                unknown_data_index = list(set(unknown_data_index) - set(compare_distance))
        if len(unknown_data_index) == 0:
            unknown_data_index = list(range(self.data.shape[0]))

        unknown_data = self.data[unknown_data_index, :]

        return unknown_data, known_data

class Cluster:

    def __init__(self, para_data):
        self.data = para_data

    def k_means(self, para_k, frequency):
        kcluster = cluster.KMeans(para_k).fit(self.data)
        temp_labels = list(kcluster.labels_)
        labels_list = []
        for i in range(len(temp_labels)):
            labels_list.append(temp_labels[i]+frequency+100)  # 给伪标签一个很大的数值，避免其与已有标签混淆
        labels_num = list(set(labels_list))
        return kcluster.labels_, labels_list, labels_num

    def dbscan(self, frequency):
        eps = self.find_optimal_eps(self.data)
        db = cluster.DBSCAN(eps=eps, min_samples=2).fit(self.data)
        temp_labels = db.labels_
        index = np.where(temp_labels != -1)[0]
        temp_labels = np.array(temp_labels)[index]
        normal_data = self.data[index]
        labels = []
        n_clusters_ = len(set(temp_labels)) - (1 if -1 in temp_labels else 0)
        n_noise_ = list(labels).count(-1)
        for i in range(len(temp_labels)):
            labels.append(temp_labels[i]+frequency+100)
        labels_num = list(set(labels))
        labels = np.array(labels).reshape(len(labels), 1)

        return normal_data, labels, labels_num
    def find_optimal_eps(self, data):
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        return np.median(distances)

class BallUpdate:

    def __init__(self, para_data):
        self.data = para_data

    def new_balls(self, para_cluster_method, para_s, frequency):

        temp_cluster = Cluster(self.data)
        if para_cluster_method == "kmeans":
            _, data_label, label_list = temp_cluster.k_means(1, frequency)
        if para_cluster_method == "dbscan":
            self.data, data_label, label_list = temp_cluster.dbscan(frequency)
        open_sv, hypersphere = Select(self.data, data_label, para_s, len(label_list),
                                        label_list, para_k=3).find_all_data_sv()
        return hypersphere, self.data, data_label, label_list, open_sv

    def knowledge_update(self, init_hypersphere, new_hypersphere):
        """
        merge all granular balls
        """
        for i in range(len(new_hypersphere["radius"])):
            init_hypersphere["radius"].append(new_hypersphere["radius"][i])
            init_hypersphere["center"].append(new_hypersphere["center"][i])
            init_hypersphere["sv"].append(new_hypersphere["sv"][i])
            init_hypersphere["sv_index"].append(new_hypersphere["sv_index"][i])
            init_hypersphere["alpha"].append(new_hypersphere["alpha"][i])

        return init_hypersphere
    def sampling_update(self, old_sv, new_sv):
        open_sampling = np.vstack((old_sv, new_sv))
        return open_sampling


def data_merge(para_init_data, para_new_data, para_init_label, para_new_label):
    data = np.vstack((para_init_data, para_new_data))
    label = np.vstack((para_init_label, para_new_label))
    return data, label


class SelectFeature:
    """
    select important features from candidate features
    """
    def __init__(self, para_select_att, para_remain_att):
        self.select_att = para_select_att
        self.remain_att = para_remain_att

    def select_feature(self, para_radius, splicing_data):
        only_data = splicing_data[:, :-1]  # 提取特征数据
        only_label = splicing_data[:, -1]  # 提取标签数据

        self.temp_label_dict = Counter(only_label)  # 统计每个标签的数量
        self.temp_label_list = list(self.temp_label_dict)  # 获取标签列表

        attribute_left = list(range(only_data.shape[1]))  # 初始化剩余特征
        attribute_select = []  # 初始化已选择特征



        min_information_values = 1  # 初始化最小信息值
        first_information_values = 0  # 初始化第一个信息值
        current_attribute_relation = np.ones((only_data.shape[0], only_data.shape[0]))  # 初始化当前特征关系矩阵
        relation_list = knsi0516.GetAttributeImportance(only_data, only_label,
                                                        para_radius).relation_matrix()  # 计算特征重要性关系矩阵

        while attribute_left:
            each_attribute_information = []

            for k in range(len(attribute_left)):  # 遍历剩余特征
                self_information = 0
                temp_lower_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)
                temp_upper_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)

                array_relation = relation_list[k] if not attribute_select else np.minimum(current_attribute_relation,
                                                                                          relation_list[k])

                for i in range(only_data.shape[0]):  # 计算上下近似集
                    neighbor_index_i = np.where(array_relation[i, :] == 1)[0]
                    neighbor_label_set = set(only_label[neighbor_index_i])
                    if len(neighbor_label_set) == 1:
                        label = list(neighbor_label_set)[0]
                        temp_lower_neighbor_dict[label] += 1
                        temp_upper_neighbor_dict[label] += 1
                    else:
                        for label in neighbor_label_set:
                            temp_upper_neighbor_dict[label] += 1

                for label in self.temp_label_list:  # 计算每个标签的精度和信息增益
                    temp_num_class_i = self.temp_label_dict[label]
                    temp_upper_approx = temp_upper_neighbor_dict[label]
                    temp_lower_approx = temp_lower_neighbor_dict[label] or (1 / temp_num_class_i)

                    precision = temp_lower_approx / temp_upper_approx
                    self_information += -(1 - precision) * np.log(precision)

                each_attribute_information.append(self_information)

            attribute_information_values = np.argsort(each_attribute_information, kind='stable')
            position = attribute_information_values[0]
            min_position_information_values = each_attribute_information[position]

            if not attribute_select:
                attribute_select.append(attribute_left[position])
                min_information_values = 1
                first_information_values = min_position_information_values  # 更新 first_information_values
            else:
                similarity = min_information_values - min_position_information_values / first_information_values
                if similarity > 0.001:
                    attribute_select.append(attribute_left[position])
                    min_information_values = min_position_information_values / first_information_values  # 更新 min_information_values
                else:
                    break

            current_attribute_relation = np.minimum(current_attribute_relation, relation_list[position])
            attribute_left.pop(position)
            relation_list.pop(position)

        remain_attributes = list(set(range(only_data.shape[1])) - set(attribute_select))  # 减1避免标签列被误包括
        return attribute_select, remain_attributes

    def select_feature2(self, para_radius, splicing_data):
        only_data = splicing_data[:, :-1]  # 提取特征数据
        only_label = splicing_data[:, -1]  # 提取标签数据

        self.temp_label_dict = Counter(only_label)  # 统计每个标签的数量
        self.temp_label_list = list(self.temp_label_dict)  # 获取标签列表

        # 已选择的特征子集
        attribute_select = self.select_att.copy()
        # 剩余的特征子集
        attribute_left = self.remain_att.copy()

        # 初始化信息量
        min_information_values = 1
        first_information_values = 0
        current_attribute_relation = np.ones((only_data.shape[0], only_data.shape[0]))  # 初始化当前特征关系矩阵
        relation_list = knsi0516.GetAttributeImportance(only_data, only_label,
                                                        para_radius).relation_matrix()  # 计算特征重要性关系矩阵

        # 如果已经有选择的特征子集，初始化first_information_values和min_information_values
        if attribute_select:
            selected_information = []
            for k in attribute_select:
                self_information = 0
                temp_lower_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)
                temp_upper_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)
                array_relation = relation_list[k]

                for i in range(only_data.shape[0]):  # 计算上下近似集
                    neighbor_index_i = np.where(array_relation[i, :] == 1)[0]
                    neighbor_label_set = set(only_label[neighbor_index_i])
                    if len(neighbor_label_set) == 1:
                        label = list(neighbor_label_set)[0]
                        temp_lower_neighbor_dict[label] += 1
                        temp_upper_neighbor_dict[label] += 1
                    else:
                        for label in neighbor_label_set:
                            temp_upper_neighbor_dict[label] += 1

                for label in self.temp_label_list:  # 计算每个标签的精度和信息增益
                    temp_num_class_i = self.temp_label_dict[label]
                    temp_upper_approx = temp_upper_neighbor_dict[label]
                    temp_lower_approx = temp_lower_neighbor_dict[label] or (1 / temp_num_class_i)

                    precision = temp_lower_approx / temp_upper_approx
                    self_information += -(1 - precision) * np.log(precision)

                selected_information.append(self_information)

            # 更新first_information_values和min_information_values
            first_information_values = selected_information[0]
            min_information_values = selected_information[-1]/first_information_values

        # 评估剩余特征子集的重要度
        while attribute_left:
            each_attribute_information = []

            for k in range(len(attribute_left)):  # 遍历剩余特征
                self_information = 0
                temp_lower_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)
                temp_upper_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)

                array_relation = np.minimum(current_attribute_relation, relation_list[attribute_left[k]])

                for i in range(only_data.shape[0]):  # 计算上下近似集
                    neighbor_index_i = np.where(array_relation[i, :] == 1)[0]
                    neighbor_label_set = set(only_label[neighbor_index_i])
                    if len(neighbor_label_set) == 1:
                        label = list(neighbor_label_set)[0]
                        temp_lower_neighbor_dict[label] += 1
                        temp_upper_neighbor_dict[label] += 1
                    else:
                        for label in neighbor_label_set:
                            temp_upper_neighbor_dict[label] += 1

                for label in self.temp_label_list:  # 计算每个标签的精度和信息增益
                    temp_num_class_i = self.temp_label_dict[label]
                    temp_upper_approx = temp_upper_neighbor_dict[label]
                    temp_lower_approx = temp_lower_neighbor_dict[label] or (1 / temp_num_class_i)

                    precision = temp_lower_approx / temp_upper_approx
                    self_information += -(1 - precision) * np.log(precision)

                each_attribute_information.append(self_information)

            attribute_information_values = np.argsort(each_attribute_information, kind='stable')
            position = attribute_information_values[0]
            min_position_information_values = each_attribute_information[position]

            similarity = min_information_values - min_position_information_values / first_information_values
            if similarity > 0.001:
                selected_attribute = attribute_left[position]
                attribute_select.append(selected_attribute)
                min_information_values = min_position_information_values / first_information_values  # 更新 min_information_values
                current_attribute_relation = np.minimum(current_attribute_relation, relation_list[selected_attribute])
            else:
                break

            attribute_left.remove(selected_attribute)  # 移除已选择的特征

        return attribute_select, attribute_left





