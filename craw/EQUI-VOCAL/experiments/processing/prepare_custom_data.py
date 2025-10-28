from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test
from src.utils import program_to_dsl

if __name__ == '__main__':
    # 定义目标查询（示例：红色物体o0在o1左侧，之后o0在o1右侧）
    query_program = [
        {
            "scene_graph": [
                {"predicate": "Color", "parameter": "red", "variables": ["o0"]},
                {"predicate": "LeftOf", "parameter": None, "variables": ["o0", "o1"]}
            ],
            "duration_constraint": 1  # 持续1个时间单位
        },
        {
            "scene_graph": [{"predicate": "RightOf", "parameter": None, "variables": ["o0", "o1"]}],
            "duration_constraint": 1
        }
    ]
    # 转换为紧凑查询字符串
    query_str = program_to_dsl(query_program)
    print("生成的查询字符串：", query_str)

    # 自定义数据集名称
    dataset_name = "my_custom_video"
    # 生成输入数据（关联数据库表Obj_custom）
    prepare_data_given_target_query(
        query_str=query_str,
        run_id=0,
        split=1,
        dataset_name=dataset_name,
        table_name="Obj_trajectories",  # 自定义数据表名
        sampling_rate=2  # 按帧率下采样（如每2帧取1帧）
    )

    # 划分训练集和测试集（500个训练样本）
    construct_train_test(
        data_dir=f"inputs/{dataset_name}",
        n_train=500
    )