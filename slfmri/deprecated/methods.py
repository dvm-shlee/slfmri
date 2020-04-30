from __future__ import print_function
import os
import pip
import shutil
import sys
import logging
import logging.handlers
import shlex
import datetime
from subprocess import PIPE, Popen


def splitnifti(path):
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def splitext(path):
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def shell(cmd, logger=None):
    """ Execute shell command
    :param cmd: str, command to execute
    :return: stdout, error
    """
    try:
        processor = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        if logger != None:
            logger.info("Shell::Success [{}]".format(cmd))
        out, err = processor.communicate()
        return out, err
    except OSError as e:
        if logger != None:
            logger.info("Shell::Error [{}]".format(e))
        return None, None


def get_logger(path, name):
    """ Logger
    :param path:
    :param name:
    :return:
    """
    today = "".join(str(datetime.date.today()).split('-'))

    # create logger
    logger = logging.getLogger('{0}'.format(name))
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(path, '{0}-{1}.log'.format(name, today)))
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger







def path_splitter(path):
    """ Split path sting into list
    :param path: path string want to split
    :type path: str
    """
    return path.strip(os.sep).split(os.sep)


def copyfile(output_path, input_path):
    """ copy files from input_path to output_path
    :param output_path: destination path
    :param input_path: origin path
    :type output_path: str
    :type input_path: str
    """
    shutil.copyfile(input_path, output_path)


