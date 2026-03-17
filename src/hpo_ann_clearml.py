from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.parameters import UniformParameterRange, DiscreteParameterRange

def job_complete_callback(job_id, objective_value, objective_iteration, job_parameters, top_performance_job_id):
    print("Job completed")
    print("job_id:", job_id)
    print("objective_value:", objective_value)
    print("objective_iteration:", objective_iteration)
    print("job_parameters:", job_parameters)
    print("top_performance_job_id:", top_performance_job_id)
    print("-" * 80)


def main():
    base_task_id="bb99aa2778fa43f39a46d30d8674d961"
    task = Task.init(
        project_name="605-Vehicle_Maintainance-project",
        task_name="ANN HPO Controller",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,    
        hyper_parameters=[
            DiscreteParameterRange("General/batch_size", values=[32, 64, 128]),
            UniformParameterRange("General/learning_rate", min_value=0.0001, max_value=0.005, step_size=0.0005),
            DiscreteParameterRange("General/optimizer", values=["adam", "rmsprop"]),
            DiscreteParameterRange("General/activation", values=["relu", "elu", "tanh"]),
            DiscreteParameterRange("General/num_layers", values=[1, 2, 3]),
            DiscreteParameterRange("General/units_layer_1", values=[64, 128, 256]),
            DiscreteParameterRange("General/units_layer_2", values=[32, 64, 128]),
            DiscreteParameterRange("General/units_layer_3", values=[16, 32, 64]),
            UniformParameterRange("General/dropout", min_value=0.1, max_value=0.5, step_size=0.1),
            UniformParameterRange("General/l2_reg", min_value=0.00001, max_value=0.001, step_size=0.0001),
            DiscreteParameterRange("General/batch_norm", values=[True, False]),
        ],
        objective_metric_title="validation",
        objective_metric_series="auc",
        objective_metric_sign="max",
        execution_queue="default",
        max_number_of_concurrent_tasks=8,
        total_max_jobs=20,
        save_top_k_tasks_only=5,
        min_iteration_per_job=10,
        max_iteration_per_job=30,
    )

    optimizer.set_report_period(1)
    optimizer.start(job_complete_callback=job_complete_callback)
    optimizer.wait()
    optimizer.stop()

    top_experiments = optimizer.get_top_experiments(top_k=5)

    print("\nTop experiments:")
    for i, exp in enumerate(top_experiments, 1):
        print(f"{i}. {exp.id} | {exp.name}")

    task.close()


if __name__ == "__main__":
    main()