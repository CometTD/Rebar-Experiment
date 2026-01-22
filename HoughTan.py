import numpy as np
import cv2


def hough_transformP2_vectorized(edges, angle, angle_range=3, rho_res=1, theta_res=np.pi / 180,
                                 threshold=50, min_line_length=50, max_line_gap=10, merge_threshold_rho=10,
                                 merge_threshold_theta=np.pi / 90, top_k=5):
    height, width = edges.shape
    max_rho = int(np.hypot(height, width))

    theta_min = np.deg2rad(angle - angle_range)
    theta_max = np.deg2rad(angle + angle_range)
    thetas = np.arange(theta_min, theta_max, theta_res)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # 获取边缘点 (y, x)
    y_idxs, x_idxs = np.nonzero(edges)
    num_thetas = len(thetas)

    # 扩展形状为 (num_points, num_thetas)
    x_mat = np.tile(x_idxs[:, None], (1, num_thetas))
    y_mat = np.tile(y_idxs[:, None], (1, num_thetas))
    cos_mat = cos_t[None, :]
    sin_mat = sin_t[None, :]

    # 计算所有组合对应的 rho
    rhos = np.round(x_mat * cos_mat + y_mat * sin_mat).astype(int) + max_rho

    accumulator = np.zeros((2 * max_rho, num_thetas), dtype=np.int32)

    # 使用矢量化批量投票
    for i in range(rhos.shape[0]):
        accumulator[rhos[i], np.arange(num_thetas)] += 1

    # 获取超过阈值的索引
    rho_idx, theta_idx = np.where(accumulator > threshold)
    votes = accumulator[rho_idx, theta_idx]

    detected_lines = []
    for r_idx, t_idx, vote in zip(rho_idx, theta_idx, votes):
        rho = r_idx - max_rho
        theta = thetas[t_idx]
        detected_lines.append((rho, theta, vote))

    # 合并相近的直线（与原函数一致）
    merged_lines = []
    for rho, theta, vote in detected_lines:
        merged = False
        for i, (mrho, mtheta, mvote) in enumerate(merged_lines):
            if abs(rho - mrho) < merge_threshold_rho and abs(theta - mtheta) < merge_threshold_theta:
                if vote > mvote:
                    merged_lines[i] = (rho, theta, vote)
                merged = True
                break
        if not merged:
            merged_lines.append((rho, theta, vote))

    merged_lines.sort(key=lambda x: x[2], reverse=True)
    merged_lines = merged_lines[:top_k]

    # 转换为实际坐标
    lines = []
    for rho, theta, _ in merged_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        if np.hypot(x2 - x1, y2 - y1) >= min_line_length:
            lines.append([(x1, y1), (x2, y2)])

    return lines

def get_main_axis_angle_pca(edge_image):
    # 提取边缘点的坐标
    points = np.column_stack(np.where(edge_image > 0))

    # 检查是否有足够的点
    if len(points) < 2:
        return None

    # 应用 PCA
    mean, eigenvectors = cv2.PCACompute(np.float32(points), mean=None, maxComponents=2)

    # 主方向的特征向量
    main_direction = eigenvectors[0]

    # 计算角度（弧度转度）
    angle = np.arctan2(main_direction[1], main_direction[0]) * 180 / np.pi
    return angle


def calculate_3d_coordinates(x, y, depth, rgb_frame, focal_length=(795.8499755859375 + 787.8486328125) / 2):
    z = depth
    f = int(focal_length)
    x_3d = (x - (1280 / 2)) * z / f
    y_3d = (y - (720 / 2)) * z / f
    return x_3d, y_3d, z


def calculate_3d(P1, P2, P3, P4):
    m1 = (P2[1] - P1[1]) / (P2[0] - P1[0]) if P2[0] != P1[0] else float('inf')
    m2 = (P4[1] - P3[1]) / (P4[0] - P3[0]) if P4[0] != P3[0] else float('inf')
    if m1 == float('inf') and m2 == float('inf'):
        return abs(P1[0] - P3[0])
    elif m1 == float('inf') or m2 == float('inf'):
        mc = m1 if m2 == float('inf') else m2
    else:
        mc = (m1 + m2) / 2
    c1 = P1[1] - mc * P1[0]
    c2 = P3[1] - mc * P3[0]

    distance = abs(c2 - c1) / np.sqrt(1 + mc ** 2)
    return distance

def calculate_spacing(L):
    L = L - (0.26 * L - 4.29)
    return L