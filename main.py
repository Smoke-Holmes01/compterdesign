from preprocess_groups import preprocess_all_groups
from render_obj_views import render_all_models
from match_groups_to_objs import build_score_matrix, hungarian_assign


def main():
    groups_root = "data/image_groups"
    model_dir = "data/models"

    preprocessed_root = "outputs/preprocessed_groups"
    render_root = "outputs/renders"
    results_root = "outputs/results"

    print("Step 1/4: preprocess all groups")
    group_info = preprocess_all_groups(groups_root, preprocessed_root, out_size=512)
    print(f"Preprocessed groups: {len(group_info)}")

    print("\nStep 2/4: render all OBJ models")
    render_info = render_all_models(
        model_dir=model_dir,
        output_root=render_root,
        azimuth_list=list(range(0, 360, 10)),
        elevation_list=[0, 10, 20, 30],
        image_size=512,
        distance=2.5
    )
    print(f"Rendered models: {len(render_info)}")

    print("\nStep 3/4: build score matrix")
    group_names, model_names, score_matrix, detailed_scores = build_score_matrix(
        preprocessed_root=preprocessed_root,
        render_root=render_root,
        output_dir=results_root
    )
    print(f"Score matrix shape: {score_matrix.shape}")

    print("\nStep 4/4: Hungarian global assignment")
    final_result = hungarian_assign(
        group_names=group_names,
        model_names=model_names,
        score_matrix=score_matrix,
        detailed_scores=detailed_scores,
        output_dir=results_root
    )

    print("\nFinal assignments:")
    for item in final_result["assignments"]:
        print(f"{item['group_name']} -> {item['assigned_model']} | score={item['score']:.4f}")

    print("\nDone.")
    print("Final result file: outputs/results/final_assignments.json")


if __name__ == "__main__":
    main()