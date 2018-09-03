import sys
import os
from flask import Flask
from flask import request
from flask import session
from flask import g
from flask import redirect
from flask import url_for
from flask import abort
from flask import render_template
from flask import flash
from flask import make_response
from flask import Markup
from flask import send_from_directory
from flask import send_file
from helper import clip_path
from helper import clip_paths

reload(sys)
sys.setdefaultencoding('utf8')

GraphBoard = Flask(__name__)
GraphBoard.config['graph_attrs'] = ""
GraphBoard.config['graph_option'] = ""
GraphBoard.config['optimized_graph_attrs'] = ""
GraphBoard.config['optimized_graph_option'] = ""
GraphBoard.config['mem_info']=""
GraphBoard.config['disable_optimization'] = bool()
GraphBoard.config['config'] = dict()
GraphBoard.config.from_object(__name__)


@GraphBoard.route('/')
def board_home(): 
    """
    doc index route
    """ 
    config = GraphBoard.config['config']
    parser_config = []
    # parsing target framework config
    framework = config.framework
    parser_config.append(["Framework", framework, " The target framework processing "])
    if framework == "CAFFE":
        protos = clip_paths(config.framework_config_dict['ProtoPaths'])
        parser_config.append(["Proto", protos, "Protobuf define files "])
        prototxt = clip_path(config.framework_config_dict['PrototxtPath'])
        parser_config.append(["Prototxt", prototxt, "Network tarits define"])
    model = clip_path(config.framework_config_dict['ModelPath'])
    parser_config.append(["Model", model, "Model parameter file"])
    return render_template('index.html', \
                            disable_optimization=GraphBoard.config['disable_optimization'], 
                            parser_config=parser_config, 
                            graph_def=GraphBoard.config['graph_option'], 
                            attrs=GraphBoard.config['graph_attrs'])


@GraphBoard.route('/optimization')
def board_optimization(): 
    """
    doc optimization route
    """ 
    config = GraphBoard.config['config']
    parser_config = []
    # parsing optimized model
    optimized_graph_path = clip_path(config.optimizedGraphPath)
    parser_config.append(["Optimization model", optimized_graph_path, \
                          "graph model optimized by anakin inference framework"])
    return render_template('optimization.html', 
                           parser_config=parser_config, 
                           graph_def=GraphBoard.config['optimized_graph_option'], 
                           attrs=GraphBoard.config['optimized_graph_attrs'],
						   mem_info=GraphBoard.config['mem_info'])


@GraphBoard.route('/<path:filename>', methods=['GET', 'POST'])
def response_local_file(filename): 
    """
    doc show local files ( png , jpg ...)
    """
    print filename
    return send_file('static/' + filename)
