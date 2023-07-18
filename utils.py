# -*- coding: utf-8 -*-
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import logging
import time
from tensorboardX import SummaryWriter


def print_and_logger(string, logger):
    print(string)
    logger.info(string)


def init_logger(args):
    """
    初始化logger
    """
    log_dir = 'log_{}'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    f = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode='w+')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    f.setFormatter(formatter)
    logger.addHandler(f)

    l_str = '> Arguments:'
    print_and_logger(l_str, logger)
    for key in args.__dict__.keys():
        l_str = '\t{}: {}'.format(key, args.__dict__[key])
        print_and_logger(l_str, logger)

    return logger, log_dir


class DictTable(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table width=100%>"]
        for key, value in self.items():
            html.append("<tr>")
            html.append("<td>{0}</td>".format(key))
            html.append("<td>{0}</td>".format(value))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


def init_logger_writer(args):
    """
    初始化logger（用于文字记录训练过程）和writer（用于可视化loss、acc等关注的值）
    """
    logger, log_dir = init_logger(args)
    writer = SummaryWriter(log_dir)
    writer.add_text('args', DictTable(args.__dict__)._repr_html_(), 0)  # record the training configurations

    return logger, writer


def inform_logger_writer(logger, writer, epoch, train_loss, time_list):
    l_str = '[epoch {}] train time: {:1.4f}s'. \
        format(epoch, time_list[1] - time_list[0])
    print_and_logger(l_str, logger)
    l_str = '***** Training set result ***** loss: {:1.4f}'.format(train_loss)
    print_and_logger(l_str, logger)

    writer.add_scalar('train/loss', train_loss, epoch)


def close_logger_writer(logger, writer, start_time):
    print_and_logger(' ', logger)
    l_str = 'Elapsed time {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
    print_and_logger(l_str, logger)

    loggers = list(logger.handlers)
    for i in loggers:
        logger.removeHandler(i)
        i.flush()
        i.close()
    writer.close()

