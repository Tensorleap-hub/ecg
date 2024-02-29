from ecg.utils.find_picks import find_picks
from ecg.utils.visualizers import *
from leap_binder import *
from tensorflow import keras


def plot_ecg(ecg_signals, sig_len):
    fs = 500
    time = np.arange(sig_len) / fs
    sig_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Plot each ECG signal separately
    for i in range(12):
        plt.figure(figsize=(10, 6))
        plt.plot(time, ecg_signals[:, i])
        plt.title('ECG Signal - ' + sig_names[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


def plot_ecg_with_picks(ecg_signals, sig_len, picks):
    fs = 500
    time = np.arange(sig_len) / fs
    sig_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Plot each ECG signal separately
    for i in range(12):
        plt.figure(figsize=(10, 6))
        plt.plot(time, ecg_signals[:, i])
        plt.scatter(picks, ecg_signals[picks], color='red')
        plt.title('ECG Signal - ' + sig_names[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


def check_custom_test(model_name):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # model_path = 'assets/patch_12x16.h5'
    # vit = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    if model_name == "VIT":
        model = tf.keras.models.load_model \
            ('/Users/chenrothschild/repo/Tensorleap-hub/internal-projects/assets/patch_12x16.h5')

    elif model_name == "CNN":
        model = tf.keras.models.load_model \
            ('/Users/chenrothschild/repo/Tensorleap-hub/ecg/ecg/test_models/cnn_new_model.h5')

    else:
        model = tf.keras.models.load_model \
            ('/Users/chenrothschild/repo/Tensorleap-hub/ecg/ecg/test_models/custom_model.h5')

    responses = preprocess_func()
    train = responses[0]
    val = responses[1]
    unlabeled_set = unlabeled_data()
    responses_set = unlabeled_set

    for idx in range(20):
        # if idx == 33:
        #     a =1

        print(f"idx: {idx}")
        # print(f"idx_f: {int(idx // CONFIG['SAMPLE_TIMES'])}")
        # get input and gt

        if model_name == "VIT":
            input = input_encoder(idx, responses_set)
            # plot_ecg(input, 224)
            concat = np.expand_dims(input, axis=0)
            y_pred = model([concat])

        else:
            input = input_encoder_raw(idx, responses_set)
            # plot_ecg(input, 5000)
            concat = np.expand_dims(input, axis=0)
            y_pred = model([concat])
            gt = gt_scp(idx, responses_set)
            gt = np.expand_dims(gt, axis=0)
            y_true = tf.convert_to_tensor(gt, dtype='float32')

        # get custom meta data
        scp_gt = get_scp_gt(idx, responses_set)
        gender = gender_metadata(idx, responses_set)
        age = age_metadata(idx, responses_set)
        height = height_metadata(idx, responses_set)
        weight = weight_metadata(idx, responses_set)
        device = device_metadata(idx, responses_set)
        nurse = nurse_metadata(idx, responses_set)
        site = site_metadata(idx, responses_set)
        validated_by_human = validated_by_human_metadata(idx, responses_set)
        gender_int = gender_metadata_int(idx, responses_set)
        scp_code_dict = scp_code_metadata_dict(idx, responses_set)
        waveform_mean = waveform_mean_metadata_dict(idx, responses_set)
        waveform_min = waveform_min_metadata_dict(idx, responses_set)
        waveform_max = waveform_max_metadata_dict(idx, responses_set)

        # loss
        ls = keras.losses.CategoricalCrossentropy()(y_true, y_pred)

        # vis
        waveform = display_waveform_I(input)
        gt_horizontal_bar = horizontal_bar_visualizer_with_labels_name(np.array(gt, dtype='float32')[0])
        pred_horizontal_bar = horizontal_bar_visualizer_with_labels_name(np.array(y_pred, dtype='float32')[0])
        graph_visuaizer_ = graph_visuaizer(input)
        graph_heatmap_visualizer_ = graph_heatmap_visualizer(input)
        img_vis_T_ = img_vis_T(input)
        heatmap_vis_T_ = heatmap_vis_T(input)

        # custom metrics
        pred_label_metadata = pred_label(gt, y_pred)

        print(f"idx: {idx} finished successfully")
    print("Custom tests finished successfully")


if __name__ == '__main__':
    model_names = ["VIT", "CNN", "custom"]
    check_custom_test(model_names[2])
