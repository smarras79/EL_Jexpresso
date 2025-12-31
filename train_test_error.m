clc;clear
close all

load('JX_NN_error.mat')
plot(train_err)
hold on
plot(test_err)
set(gca, 'YScale', 'log')
legend('train error','test error')

