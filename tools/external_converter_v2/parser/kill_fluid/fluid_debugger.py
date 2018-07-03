from ..proto import *
from ..graph_io import *
import copy
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.core import VarDesc, AttrType


class Fluid_debugger:

	def var_names_of_fetch(self, fetch_targets):
		var_names_list = []
		for var in fetch_targets:
			var_names_list.append(var.name)
		return var_names_list

	def fetch_tmp_vars(self, block, fetch_targets, var_names_list = None):
		fetch_var = block.var('fetch')
		old_fetch_names = self.var_names_of_fetch(fetch_targets)
		new_fetch_vars = []
		for var_name in old_fetch_names:
			var = block.var(var_name)
			new_fetch_vars.append(var)
		i = len(new_fetch_vars)
		if var_names_list is None:
			var_names_list = block.vars.keys()
		for var_name in var_names_list:
			if '.tmp_' in var_name and var_name not in old_fetch_names:
				var = block.var(var_name)
				new_fetch_vars.append(var)
				block.append_op(
					type='fetch',
					inputs={'X': [var_name]},
					outputs={'Out': [fetch_var]},
					attrs={'col': i})
				i = i + 1
		return new_fetch_vars
