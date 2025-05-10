import open3d as o3d
import numpy as np
import random
import os

def load_obj(obj_file_path):
    vertices = []
    faces = []
    face_colors = []
    object_names = []  # 存储每个对象的名称
    object_face_map = {}  # 映射对象名到其关联的面
    current_material = None
    current_object = None

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('o '):  # 对象名称行
                current_object = line.strip().split()[1]
                object_names.append(current_object)
                object_face_map[current_object] = []
            elif line.startswith('v '):  # 处理顶点
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):  # 处理面
                parts = line.strip().split()
                face = []
                for p in parts[1:]:
                    indices = p.split('/')
                    try:
                        vertex_index = int(indices[0]) - 1  # 顶点索引减1
                        face.append(vertex_index)
                    except ValueError:
                        continue
                if len(face) == 3:
                    faces.append(face)
                    if current_material:
                        face_colors.append(current_material)
                    if current_object:
                        object_face_map[current_object].append(len(faces) - 1)
            elif line.startswith('usemtl '):  # 材质信息
                material_str = line.strip().split()[1]
                color_parts = material_str[1:].split(',')
                r, g, b = map(float, [color_parts[0][1:], color_parts[1][1:], color_parts[2][1:]])
                current_material = (int(r * 255), int(g * 255), int(b * 255))

    return np.array(vertices), np.array(faces), face_colors, object_names, object_face_map

def sample_points_on_faces(vertices, faces, face_colors, num_samples):
    def calculate_area(v1, v2, v3):
        return np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2

    areas = [calculate_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in faces]
    total_area = sum(areas)
    samples_per_face = [int(num_samples * (area / total_area)) for area in areas]

    sampled_points = []
    sampled_colors = []

    for idx, (face, num_points) in enumerate(zip(faces, samples_per_face)):
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        color = face_colors[idx] if idx < len(face_colors) else (128, 128, 128)  # 默认灰色
        for _ in range(num_points):
            r1, r2 = random.random(), random.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            point = (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3
            sampled_points.append(point)
            sampled_colors.append(color)

    return np.array(sampled_points), np.array(sampled_colors)

def save_point_cloud(filename, points, colors, categories=None):
    if categories is not None:
        # 确保类别为字符串类型，并调整形状
        categories = np.array(categories).reshape(-1, 1)
        # 将坐标、颜色和类别拼接
        data = np.hstack((points, colors, categories))
    else:
        # 仅包含坐标和颜色
        data = np.hstack((points, colors))

    # 确保数据类型一致
    structured_data = []
    for row in data:
        # 将 XYZ 转为浮点数，RGB 转为整数，类别转为字符串
        structured_row = [
            float(row[0]), float(row[1]), float(row[2]),  # XYZ
            int(row[3]), int(row[4]), int(row[5]),  # RGB
            str(row[6]) if len(row) > 6 else ""  # 类别
        ]
        structured_data.append(structured_row)

    # 转换为 Numpy 数组
    structured_data = np.array(structured_data, dtype=object)

    # 保存文件
    np.savetxt(filename, structured_data, fmt='%f %f %f %d %d %d %s')

def main():
    base_path = 'E:/pythonfile_1/Instance/Revit_shili'
    for i in range(1, 11):
        obj_file_path = f'{base_path}/object{i}/construction{i}_processed.obj'
        output_dir = f'{base_path}/object{i}/sampled_point_clouds'
        os.makedirs(output_dir, exist_ok=True)

        vertices, faces, face_colors, object_names, object_face_map = load_obj(obj_file_path)

        total_sampled_points = []
        total_sampled_colors = []
        total_sampled_categories = []

        for obj_name in object_names:
            obj_faces = [faces[idx] for idx in object_face_map[obj_name]]
            obj_colors = [face_colors[idx] for idx in object_face_map[obj_name] if idx < len(face_colors)]
            obj_colors = obj_colors or [(128, 128, 128)] * len(obj_faces)  # 默认灰色

            num_samples = 100000  # 每个对象采样点数
            sampled_points, sampled_colors = sample_points_on_faces(vertices, obj_faces, obj_colors, num_samples)

            total_sampled_points.extend(sampled_points)
            total_sampled_colors.extend(sampled_colors)
            total_sampled_categories.extend([obj_name] * len(sampled_points))  # 添加类别名称

            # 按类别保存点云文件
            save_point_cloud(os.path.join(output_dir, f"{obj_name}.txt"), sampled_points, sampled_colors, [obj_name] * len(sampled_points))

        # 保存所有采样的点云为一个文件
        save_point_cloud(os.path.join(output_dir, "all_sampled_points.txt"), total_sampled_points, total_sampled_colors, total_sampled_categories)

        # 可视化
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(total_sampled_points)
        #pcd.colors = o3d.utility.Vector3dVector(np.array(total_sampled_colors) / 255.0)
        #o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
