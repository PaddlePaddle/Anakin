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
var MemoryChart = echarts.init(document.getElementById('memory_bar'));
var memory_bar_colors = ['#50AE28', '#89CD6B', '#1B5C00', '#398D6B'];
var memory_option = {
	color: memory_bar_colors,
    tooltip : {
        trigger: 'axis'
    },
	legend: {
        data: ['System Used', 'Model Used', 'Temp Space', 'Sum Used']
    },
    toolbox: {
        show : true,
        feature : {
            dataView : {show: true, readOnly: false},
            magicType : {show: true, type: ['line', 'bar']},
            restore : {show: true},
            saveAsImage : {show: true}
        }
    },
    calculable : true,
	grid: {
		show: false,
	},
    xAxis : [
        {
            type : 'category',
			name : 'Version',
            data : ['TensorRT','anakin_v2'],
			axisLine: {
                lineStyle: {
                    color: memory_bar_colors[0]
                }
            },
        }
    ],
    yAxis : [
        {
            type : 'value',
			name : 'MB',
			axisLabel: {
                formatter: '{value} MB'
            },
			splitLine: {show: false},
			axisLine: {
                lineStyle: {
                    color: memory_bar_colors[0]
                }
            }

        }
    ],
    series : [
		{
			name: 'System Used',
			type: 'bar',
			stack: 'Sum Used',
			barWidth: '15%',
			data:[1297, 371],
			label: {
                normal: {
                    show: true,
                    position: 'inside'
                }
            },
		},
        {
            name:'Model Used',
            type:'bar',
			stack: 'Sum Used',
			barWidth: '15%',
            data:[0, 52],
			label: {
                normal: {
                    show: true,
                    position: 'inside'
                }
            },
        },
		{
            name:'Temp Space',
            type:'bar',
			stack: 'Sum Used',
			barWidth: '15%',
            data:[0, 38],
			label: {
                normal: {
                    show: true,
                    position: 'inside'
                }
            },
        },
		{
			name: 'Sum Used',
            type: 'line',
            data:[1297, 461], 
			barWidth: '5%',
			symbol: 'circle',
			lineStyle: {
				normal: {
					width: 1,
					type: 'dashed',
				}
			},
			markPoint: {
                data: [
                    {type: 'max'}, {type: 'min'}
                ]
            },
		}
	]
};

MemoryChart.setOption(memory_option);

