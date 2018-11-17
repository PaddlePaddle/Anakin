#include "framework/graph/llvm/optimizer/memory_scheduler.h"

namespace anakin {

namespace graph {

void IOBlockResource::reg_self_lock_tree(io& io_in, std::vector<io>& io_out) {
    if (_self_lock_next_tree.count(io_in) <= 0) {
        _self_lock_next_tree[io_in] = io_out;
    } else {
        LOG(FATAL) << "io(" << io_in.name << ") shouldn't be registered.";
    }
}

void IOBlockResource::rm_self_lock_tree(io& io_in) {
    for (auto it = _self_lock_next_tree.begin(); it != _self_lock_next_tree.end();) {
        auto key = it->first;
        auto& io_vec = it->second;

        for (auto it_value = io_vec.begin(); it_value != io_vec.end();) {
            if (*it_value == io_in) {
                it_value = io_vec.erase(it_value);
            } else {
                ++it_value;
            }
        }

        ++it;
    }
}

void IOBlockResource::free_self(std::vector<io>& self_shared_edges, VGraph* vgraph_p) {
    for (auto& io : self_shared_edges) {
        rm_self_lock_tree(io);
    }

    for (auto it = _self_lock.begin(); it != _self_lock.end();) {
        if (_self_lock_next_tree.count(*it) <= 0) {
            LOG(FATAL) << "io(" << it->name << ") must have been registered.";
        } else {
            if (_self_lock_next_tree[*it].size() == 0) {
                //_free.push(*it);
                push_free(*it, vgraph_p);
                it = _lock.erase(it);
            } else {
                ++it;
            }
        }
    }
}

bool IOBlockResource::is_same_target(io& one, io& two, VGraph* vgraph_p) {
    io unshared_one = one;
    io unshared_two = two;
    auto find_unshared_io_one = [&](Arc<std::string, io>& arc) {
        auto share_from = unshared_one.share_from;

        if (arc.weight().name == share_from) {
            unshared_one = arc.weight();
            return Status::EXIT(" Find the matched target arc io. ");
        }

        return Status::OK();
    };

    auto find_unshared_io_two = [&](Arc<std::string, io>& arc) {
        auto share_from = unshared_two.share_from;

        if (arc.weight().name == share_from) {
            unshared_two = arc.weight();
            return Status::EXIT(" Find the matched target arc io. ");
        }

        return Status::OK();
    };

    while (unshared_one.shared) {
        vgraph_p->Scanner->BFS_Edge(find_unshared_io_one);
    }

    while (unshared_two.shared) {
        vgraph_p->Scanner->BFS_Edge(find_unshared_io_two);
    }

    if (unshared_one == unshared_two) {
        return true;
    }

    return false;
}

void IOBlockResource::push_free(io& io_free, VGraph* vgraph_p) {
    bool io_free_have_regist = false;

    for (auto it = _free.begin(); it != _free.end();) {
        if (is_same_target(*it, io_free, vgraph_p)) {
            io_free_have_regist = true;
        }

        ++it;
    }

    if (!io_free_have_regist) {
        _free.push_back(io_free);
    }
}

void IOBlockResource::free(std::vector<io>& io_vec, VGraph* vgraph_p) {
    for (auto& io_res : io_vec) {
        for (auto it = _lock.begin(); it != _lock.end();) {
            io tmp_io;
            tmp_io.name = io_res.name;

            if ((*it) == tmp_io) {
                push_free(*it, vgraph_p);
                it = _lock.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void IOBlockResource::lock(std::vector<io>& io_vec) {
    for (auto& io_res : io_vec) {
        
        if (has_free(io_res)) {
            auto tmp_io =  get_free(io_res);// get active resouce
            io_res.shared = true;

            if (tmp_io.shared) {
                io_res.share_from = tmp_io.share_from;
            } else {
                io_res.share_from = tmp_io.name;
            }

            _lock.push_back(io_res);
        } else { // alloc new io block
            io_res.shared = false;
            _lock.push_back(io_res);
        }
    }
}

bool IOBlockResource::is_locked(io& io_in) {
	for(auto it = _lock.begin(); it != _lock.end();) {
		if((*it) == io_in) {
			return true;
		} else {
			++it;
		}
	}
    return false;
}

void IOBlockResource::map_ios_to_vgraph(std::vector<io>& io_vec, VGraph* vgraph_p) {
    for (auto& io_res : io_vec) {
        auto replace_arc = [&](Arc<std::string, io>& arc) {
            if (arc.weight() == io_res) {
                auto& io_tmp = arc.weight();
                io_tmp = io_res;
                return Status::EXIT(" Find the matched target arc io. ");
            }

            return Status::OK();
        };
        vgraph_p->Scanner->BFS_Edge(replace_arc);
    }
}

void MemoryScheduler::launch(node& node_arg) {
    this->exe_push(node_arg);
    auto& node_arc_out_its = _vgraph->get_out_arc_its(node_arg.name);
    std::vector<io> io_out;
    std::vector<std::string> next_type;

    for (auto& arc_it : node_arc_out_its) {
        
            io_out.push_back(arc_it->weight());
            next_type.push_back((*_vgraph)[arc_it->top()].opName);
    }

    this->free(io_out);

    // used for memory analysis
    set_fix_io(io_out);

    if (_need_self_shared(node_arg)) {
		auto& node_arc_in_its = _vgraph->get_in_arc_its(node_arg.name);
		if(node_arc_in_its.size() > 1) {
			int selected = 0;
			std::vector<int> io_locked_idx;
			for(int i=0; i < node_arc_in_its.size(); i++) {
				//if(_io_block_res.is_locked(node_arc_in_its[i]->weight())) {
					io_locked_idx.push_back(i);
				//}
			}
			// collect all locked io bottom node's inputs io
			std::vector<io> all_collected;
			for(auto idx : io_locked_idx) {
				auto& arc_select = node_arc_in_its[idx];
				auto& temp_arc_in_its = _vgraph->get_in_arc_its(arc_select->bottom());
				for(auto& it : temp_arc_in_its) {
					all_collected.push_back(it->weight());
				}
			}
			for(auto idx : io_locked_idx) {
				bool dismiss = false;
				for(auto& io : all_collected) {
					if(node_arc_in_its[idx]->weight().shared) {
						auto& node_btm = (*_vgraph)[node_arc_in_its[idx]->bottom()]; 
						if(_need_self_shared(node_btm)) { 
							dismiss = false; 
							break; 
						}
						if((io.share_from == node_arc_in_its[idx]->weight().share_from) || \
								(io.name == node_arc_in_its[idx]->weight().share_from)) {
							dismiss = true;
							break;
						}
					} else {
						dismiss = false;
						break;
					}
				}
				if(!dismiss) {
					selected = idx;
					break;
				}
			}
            //if last op is self_shared and we need set selected to this idx
            for (auto idx: io_locked_idx){
                auto& node_btm = (*_vgraph)[node_arc_in_its[idx]->bottom()];
                if (_need_self_shared(node_btm)){
                    selected = idx;
                }
            }
			_io_block_res.push_self_lock(node_arc_in_its[selected]->weight());
			for(int i=0; i<node_arc_in_its.size(); i++) {
				if(i != selected) {
					io_out.push_back(node_arc_in_its[i]->weight());
				}
			}
			for(auto& io_tmp : io_out) {
				io_tmp.shared = true;
				if (node_arc_in_its[selected]->weight().shared) {
                	io_tmp.share_from = node_arc_in_its[selected]->weight().share_from;
            	} else {
                	io_tmp.share_from = node_arc_in_its[selected]->weight().name;
            	}
			}
			_io_block_res.reg_self_lock_tree(node_arc_in_its[selected]->weight(), io_out); 
			_io_block_res.map_ios_to_vgraph(io_out, _vgraph); // map changes to _vgraph
		} else {
			// original impl
			auto& node_arc_in_its = _vgraph->get_in_arc_its(node_arg.name);
        	CHECK_EQ(node_arc_in_its.size(),
        	         1) << "Self shared node(" << node_arg.name << ")'s input size should be 1";

        	for (auto& arc_it : node_arc_in_its) {
        	    _io_block_res.push_self_lock(arc_it->weight());
        	}
        	for (auto& io_tmp : io_out) {
        	    io_tmp.shared = true;
        	    if (node_arc_in_its[0]->weight().shared) {
        	        io_tmp.share_from = node_arc_in_its[0]->weight().share_from;
        	    } else {
        	        io_tmp.share_from = node_arc_in_its[0]->weight().name;
        	    }
        	}
			_io_block_res.reg_self_lock_tree(node_arc_in_its[0]->weight(), io_out); 
			_io_block_res.map_ios_to_vgraph(io_out, _vgraph); // map changes to _vgraph
		}
    } else {
        _io_block_res.lock(io_out); // lock out

        for (int i=0; i<io_out.size(); ++i) {
            if (next_type[i] == "ConvEltwise"){
                io_out[i].shared = false;
            }
        }

        _io_block_res.map_ios_to_vgraph(io_out, _vgraph); // map changes to _vgraph
        auto node_arc_in_its = _vgraph->get_in_arc_its(node_arg.name);
        std::vector<io> io_in;

        for (auto& arc_it : node_arc_in_its) {
            io_in.push_back(arc_it->weight());
        }

        if (node_arg.opName != "Output") {
            _io_block_res.free(io_in, _vgraph);
        }

        std::vector<io> self_shared_edges;

        if (_need_self_shared.last_op_is_self_shared(_vgraph, node_arg, self_shared_edges)) {
            _io_block_res.free_self(self_shared_edges, _vgraph);
        }
    }
}

void MemoryScheduler::set_fix_io(std::vector<io>& io_vec) {
    for (auto it = io_vec.begin(); it != io_vec.end();) {
        if (this->is_fixed(*it)) {
            it->shared = false; // doesn't shared from and doesn't shared
            it = io_vec.erase(it);
        } else {
            ++it;
        }
    }
}

} /* namespace graph */

} /* namespace anakin */


