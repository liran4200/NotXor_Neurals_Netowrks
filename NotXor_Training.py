####################################################
#  This project creates logistic neural network    #
#  implementation using Tensorflow.                #
#  Do 8(2^8) experiment with different parameters  #
#  - learning rate 0.1, 0.01                       #
#  - number of hidden nodes 4, 2                   #
#  - with bridge and without bridge                #
#  And computes NotXor(IFF) function with training,#
#  Using Gradient Descent and Cross Entropy loss   #
####################################################

import tensorflow as tf
import numpy as np

def xnor_training(hyper_parameters,):
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[1], [0], [0], [1]]
    nb_hbridge = 0
    number_of_epocs = 0
    is_all_epocs_made = True

    if hyper_parameters['short_cut']:
        nb_hbridge = hyper_parameters['nb_hidden'] + hyper_parameters['dim']  # Bridge inputs to output (highway)
    else:
        nb_hbridge = hyper_parameters['nb_hidden']

    x = tf.placeholder(tf.float32, [None, hyper_parameters['dim']]) # define input placeholders and variables
    t = tf.placeholder(tf.float32, [None, 1])
    w1 = tf.Variable(tf.random_uniform([hyper_parameters['dim'], hyper_parameters['nb_hidden']], -1, 1, seed=0)) # random weights
    w2 = tf.Variable(tf.random_uniform([nb_hbridge, hyper_parameters['nb_outputs']], -1, 1))
    b1 = tf.Variable(tf.zeros([hyper_parameters['nb_hidden']]))
    # biases are zeros (not random)
    b2 = tf.Variable(tf.zeros([hyper_parameters['nb_outputs']]))

    z1 = tf.matmul(x, w1) + b1  # Network (grah) definition
    hlayer1 = tf.sigmoid(z1 / hyper_parameters['temp'])
    if hyper_parameters['short_cut']:
        hlayer1 = tf.concat([hlayer1, x], 1)
    z2 = tf.matmul(hlayer1, w2) + b2
    out = tf.sigmoid(z2 / hyper_parameters['temp'])

    loss = -tf.reduce_sum(t * tf.log(out) + (1 - t) * tf.log(1 - out))  # Xross Entropy Loss
    optimizer = tf.train.GradientDescentOptimizer(hyper_parameters['learning_rate'])  # Grad Descent Optimizer
    train = optimizer.minimize(loss)  # training is running optimizer to minimize loss

    # training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    epocs_lost = []
    curr_loss = 0
    for i in range(hyper_parameters['nb_epocs']):  # training iterations
        curr_train, curr_loss = sess.run([train, loss],{x: x_train, t: y_train})

        if i > hyper_parameters['validation_samples']:
            validation_loss_sum_delta = 0
            for loss1, loss2 in zip(epocs_lost[-10:], epocs_lost[-9:]):
                validation_loss_sum_delta += abs(loss1 - loss2)
            if validation_loss_sum_delta < (hyper_parameters['validation_aggregate_loss_threshold']* hyper_parameters['validation_samples']) and curr_loss < hyper_parameters['validation_loss_threshold']:
                number_of_epocs = i
                is_all_epocs_made = False
                break
    if is_all_epocs_made:
        return False, 0, 0, 0

    # evaluate test accuracy
    x_test = [[0, 0], [0, 1], [1, 0], [1, 1], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    t_test = [[1], [0], [0], [1], [1], [0], [0], [1]]
    curr_out, curr_eval_loss = sess.run([out, loss], {x: x_test, t: t_test})
    return True, number_of_epocs, curr_eval_loss, curr_loss


def xnor_experiment(learning_rate, nb_hidden, short_cut, experiment_number):
    number_of_runs = 10
    dim = 2
    nb_outputs = 1
    nb_epocs = 40000
    temp = 1
    hyper_parameters = {"dim": dim,
                        "nb_outputs": nb_outputs,
                        "nb_hidden": nb_hidden,
                        "learning_rate": learning_rate,
                        "nb_epocs": nb_epocs,
                        "temp": temp,
                        "short_cut": short_cut,
                        "validation_loss_threshold": 0.2,
                        "validation_aggregate_loss_threshold": 0.0001,
                        "validation_samples": 10}
    training_loss = []
    evaluate_loss = []
    number_of_epocs_list = []
    number_of_failed = 0
    for index in range(number_of_runs):
        is_success, number_of_epocs, evaluate_test_lose, loss = xnor_training(hyper_parameters)
        if is_success:
            training_loss.append(loss)
            evaluate_loss.append(evaluate_test_lose)
            number_of_epocs_list.append(number_of_epocs)
        else:
            number_of_failed += 1
    avg_training_loss, avg_evaluate_loss, avg_number_of_epocs = calc_avg(training_loss, evaluate_loss, number_of_epocs_list)
    results = {"avg_training_loss": avg_training_loss,
               "avg_evaluate_loss": avg_evaluate_loss,
               "avg_number_of_epocs": avg_number_of_epocs,
               "std_training_loss": np.std(training_loss),
               "std_evaluate_loss": np.std(evaluate_loss),
               "std_number_of_epocs": np.std(number_of_epocs_list),
               "number_of_failed": number_of_failed}
    write_results_to_file("experiment.txt", results, experiment_number, hyper_parameters)


def calc_avg(training_loss, evaluate_loss, number_of_epocs_list):
    avg_training_loss = sum(training_loss)/len(training_loss)
    avg_evaluate_loss = sum(evaluate_loss)/len(evaluate_loss)
    avg_number_of_epocs = sum(number_of_epocs_list)/len(number_of_epocs_list)
    return avg_training_loss, avg_evaluate_loss, avg_number_of_epocs


def write_results_to_file(filename, results, exp_number, hyper_parameters):
    if exp_number == 1:
        file = open(filename, "w")
    else:
        file = open(filename, "a")
    file.write("Experiment #%d\n" % exp_number)
    #Write to file parameters for the experiment
    file.write("Hyper parameters:\n Learning rate = %f, Nb hidden = %d, Shortcut = %r\n" %
               (hyper_parameters["learning_rate"], hyper_parameters["nb_hidden"], hyper_parameters["short_cut"]))

    # Write to file results of avg
    file.write("Averages:\n Avg training loss = %f, Avg evaluate loss = %f, Avg number of epocs = %f\n" %
               (results["avg_training_loss"], results["avg_evaluate_loss"], results["avg_number_of_epocs"]))

    # Write to file results of standard deviation
    file.write("Standard deviation:\n Std training loss = %f, Std evaluate_loss = %f, Std number of epocs = %f\n" %
               (results["std_training_loss"], results["std_evaluate_loss"], results["std_number_of_epocs"]))

    file.write("Failed = %d\n\n" % results["number_of_failed"])
    file.close()


if __name__ == '__main__':
    experiment_number = 1
    learning_rates_list = [0.1, 0.01]
    nb_hidden_list = [2, 4]
    short_cuts_list = [False, True]

    for learning_rate in learning_rates_list:
        for nb_hidden in nb_hidden_list:
            for short_cut in short_cuts_list:
                xnor_experiment(learning_rate, nb_hidden, short_cut, experiment_number)
                experiment_number += 1

