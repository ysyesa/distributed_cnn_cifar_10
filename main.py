from model_loader import *
from data_loader import *
from os import listdir
import tensorflow as tf
import logging
import os
import json
import requests
import time
import gc


def get_environment_variables():
    env = {}
    try:
        env["TFJOB_NUM_DATA"] = os.environ["TFJOB_NUM_DATA"]
        env["TFJOB_TOTAL_EPOCH"] = os.environ["TFJOB_TOTAL_EPOCH"]
        env["TFJOB_CURRENT_EPOCH"] = os.environ["TFJOB_CURRENT_EPOCH"]
        return env
    except KeyError:
        raise Exception("Environment variable not satisfied.")


class Main:
    DATA_PATH = "/app/cifar_10/"
    SAVE_PATH = "/app/tensorboard/"

    def __init__(self):
        pass

    # Parsing environment variables to get constants
    environment_variables = get_environment_variables()
    NUM_DATA_LOADED = int(environment_variables["TFJOB_NUM_DATA"])
    TOTAL_EPOCH = int(environment_variables["TFJOB_TOTAL_EPOCH"])
    CURRENT_EPOCH = int(environment_variables["TFJOB_CURRENT_EPOCH"])

    TIME_GLOBAL_START = 0

    @staticmethod
    def train(server, cluster_spec, task):
        # Construct the graph and create a saver object
        with tf.Graph().as_default():

            ALL_TRAIN_X, ALL_TRAIN_Y = DataLoader.load_train_data(data_path=Main.DATA_PATH,
                                                                  num_data=Main.NUM_DATA_LOADED)
            num_training_data_per_worker = int(Main.NUM_DATA_LOADED / len(cluster_spec["worker"]))
            first_index = num_training_data_per_worker * task["index"]
            last_index = first_index + num_training_data_per_worker
            TRAIN_X, TRAIN_Y = ALL_TRAIN_X[first_index:last_index], ALL_TRAIN_Y[first_index:last_index]

            del ALL_TRAIN_X
            del ALL_TRAIN_Y
            gc.collect()

            BATCH_SIZE = 128
            TRAINING_STEP_PER_WORKER = int(num_training_data_per_worker / BATCH_SIZE)
            TRAINING_STEP_GLOBAL = Main.CURRENT_EPOCH * int(TRAINING_STEP_PER_WORKER * len(cluster_spec["worker"]))

            # Initializing model
            tf.set_random_seed(21)
            x, y, output, y_pred_cls, learning_rate = ModelLoader.load_model()
            global_step = tf.train.get_or_create_global_step()

            # Initializing loss and optimizer
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                .minimize(loss, global_step=global_step)

            # Initializing prediction and accuracy calculation
            correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            hooks = [tf.train.StopAtStepHook(last_step=TRAINING_STEP_GLOBAL)]

            steps_time = []
            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(task["index"] == 0),
                    checkpoint_dir=Main.SAVE_PATH,
                    hooks=hooks,
            ) as mon_sess:
                time_start = time.time()
                CURRENT_STEP_WORKER = 0
                while not mon_sess.should_stop():
                    if CURRENT_STEP_WORKER >= TRAINING_STEP_PER_WORKER:
                        CURRENT_STEP_WORKER = TRAINING_STEP_PER_WORKER - 1
                    BATCH_TRAIN_X = TRAIN_X[BATCH_SIZE * CURRENT_STEP_WORKER: (BATCH_SIZE + 1) * CURRENT_STEP_WORKER]
                    BATCH_TRAIN_Y = TRAIN_Y[BATCH_SIZE * CURRENT_STEP_WORKER: (BATCH_SIZE + 1) * CURRENT_STEP_WORKER]

                    step_start_time = time.time()
                    i_global, _, batch_loss, batch_acc = mon_sess.run(
                        [global_step, optimizer, loss, accuracy],
                        feed_dict={
                            x: BATCH_TRAIN_X,
                            y: BATCH_TRAIN_Y,
                            learning_rate: ModelLoader.get_learning_rate(Main.CURRENT_EPOCH)
                        }
                    )
                    step_duration = time.time() - step_start_time
                    steps_time.append(step_duration)

                    CURRENT_STEP_WORKER = CURRENT_STEP_WORKER + 1

                duration = time.time() - time_start
                print "CURRENT_STEP_WORKER: " + str(CURRENT_STEP_WORKER)
                print "TIME TAKEN for training: " + str(duration) + " s"

            del TRAIN_X
            del TRAIN_Y
            gc.collect()

            step_time_total = 0
            for step_time in steps_time:
                step_time_total = step_time_total + step_time
            step_time_average = step_time_total / len(steps_time)

            if task["index"] == 0:
                TEST_X, TEST_Y = DataLoader.load_test_data(data_path=Main.DATA_PATH, num_data=5120)
                sess = tf.Session()
                saver = tf.train.Saver()

                files = listdir(Main.SAVE_PATH)
                for index in range(len(files)):
                    files[index] = files[index].split(".")
                max_global_step = 0
                for fi in files:
                    if "model" in fi:
                        temp = int(fi[1].split("-")[1])
                        if temp > max_global_step:
                            max_global_step = temp

                saver.restore(sess, Main.SAVE_PATH + "model.ckpt-" + str(max_global_step))

                predicted_class = sess.run(
                    y_pred_cls,
                    feed_dict={x: TEST_X, y: TEST_Y,
                               learning_rate: ModelLoader.get_learning_rate(Main.CURRENT_EPOCH)}
                )
                correct = (np.argmax(TEST_Y, axis=1) == predicted_class)
                acc = correct.mean() * 100
                print "ACCURACY for training BY LOADING: " + str(acc)

                requests.post("http://10.148.0.14:5000/modify", data={
                    "tfjob_meta_name": "cnncifar10epoch" + str(Main.CURRENT_EPOCH),
                    "tfjob_current_epoch": str(Main.CURRENT_EPOCH),
                    "tfjob_current_epoch_accuracy": str(acc),
                    "tfjob_current_epoch_time": str(duration),
                    "tfjob_current_epoch_step_time": str(step_time_average),
                    "tfjob_start_time": str(Main.TIME_GLOBAL_START),
                    "tfjob_end_time": str(time.time())
                })

    @staticmethod
    def run():
        logging.info("Tensorflow version: %s", tf.__version__)
        logging.info("Tensorflow git version: %s", tf.__git_version__)

        tf_config_json = os.environ.get("TF_CONFIG", "{}")
        tf_config = json.loads(tf_config_json)
        task = tf_config.get("task", {})
        cluster_spec = tf_config.get("cluster", {})

        server = None
        if cluster_spec:
            cluster_spec_object = tf.train.ClusterSpec(cluster_spec)
            server_def = tf.train.ServerDef(
                cluster=cluster_spec_object.as_cluster_def(),
                protocol="grpc",
                job_name=task["type"],
                task_index=task["index"])

            # Create and start a server for the local task.
            server = tf.train.Server(server_def)

            # Assigns ops to the local worker by default.
            device_func = tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % server_def.task_index,
                cluster=server_def.cluster)
        else:
            device_func = tf.train.replica_device_setter()

        job_type = task.get("type", "").lower()
        if job_type == "ps":
            logging.info("Running PS code.")
            server.join()
        elif job_type == "worker":
            logging.info("Running Worker code.")
            with tf.device(device_func):
                Main.train(server=server, cluster_spec=cluster_spec, task=task)


if __name__ == "__main__":
    main = Main()
    Main.TIME_GLOBAL_START = time.time()
    main.run()
