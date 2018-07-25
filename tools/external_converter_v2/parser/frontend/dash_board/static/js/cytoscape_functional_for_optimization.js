// set edge traits
cy_graph.filter(function(element, i){
	if( element.isEdge() ){ 
		//console.log( 'edge: '+ element.data('edge_name') +" >> "+ element.data('memory_id') );
		if(element.data('shared')) {
			element.style({
				'line-color': element.data('edge_color'),
				'target-arrow-color': element.data('edge_color'),
				'label': element.data('memory_id'),
				'text-valign': 'center',
				'font-family': 'sans-serif',
				'font-size': 20, 
				'color': 'white', 
				'text-outline-width': 4, 
				'text-outline-color': element.data('edge_color'), 
			});
		} else { /* edge tensor hold real memory */
			element.style({
				'line-color': element.data('edge_color'),
				'target-arrow-color': element.data('edge_color'),
				'label': 'New' + element.data('memory_id'),
				'text-valign': 'center',
				'font-family': 'sans-serif',
				'font-size': 25, 
				'color': 'white', 
				'text-outline-width': 4, 
				'text-outline-color': element.data('edge_color'), 
			});
		}
	} 
});

// memory bar draw
var MemoryChart = echarts.init(document.getElementById('memory_bar'), 'default');
memory_option = {
    title : {
        text: 'Anakin memory optimization result',
        subtext: 'Total memory size ('+ mem_info.total_mem + 'MB)',
        x:'center'
    },
    tooltip : {
        trigger: 'item',
        formatter: "{a} <br/>{b} : {c} MB ({d}%)"
    },
    legend: {
        //orient : 'vertical',
		orient: 'horizontal',
        x : 'center',
		y : 'bottom',
        data:['temp_mem','system_mem','model_mem']
    },
    toolbox: { 
		show : true, 
		feature : { 
			dataView : {show: true, readOnly: false}, 
			restore : {show: true}, 
			saveAsImage : {show: true} 
		} 
	},	
    calculable : false,
    series : [
        {
            name:'memory type',
            type:'pie',
			selectedMode: 'single',
            radius : '55%',
            center: ['50%', '60%'],
            data:[
                {
					value: mem_info.temp_mem, 
					name:'temp_mem',
					itemStyle: {
						color: '#89CD6B'
					},
					selected:true
				},
                {
					value: mem_info.system_mem, 
					name:'system_mem',
					itemStyle: {
						color: '#1B5C00'
					}
				},
                {
					value: mem_info.model_mem, 
					name:'model_mem',
					itemStyle: {
						color: '#50AE28'
					}
				},
            ]
        }
    ]
};
                    

MemoryChart.setOption(memory_option);

