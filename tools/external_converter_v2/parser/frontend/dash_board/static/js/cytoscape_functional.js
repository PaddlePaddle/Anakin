// download view
$(document).ready(function(){ 
	$('#download').click(function(){ 
		var view_png64 = cy_graph.png();
		//$('#png-eg').attr('src', png64);
		var link = document.createElement('a');
		link.src = "https://www.baidu.com/";
		link.click();
	}); 
});

function  download() {                 
	var view_png64 = cy_graph.png({
		'output': 'blob',
		'full': 'true',
		'bg': '#f8f8f8',
	});
	var url = window.URL.createObjectURL(view_png64);
	var link = document.createElement('a');
	link.href=url;
	link.download='view.png';
	link.click();
}

// set op traits
cy_graph.nodes('node[name = "Input"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#8A8604',
		'text-outline-color': '#8A8604',
	}); 
	//console.log( ele.id() );
});
cy_graph.nodes('node[name = "Dot"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#5C4A72',
		'text-outline-color': '#5C4A72',
	}); 
});
cy_graph.nodes('node[name = "Eltwise"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#0ABDA0',
		'text-outline-color': '#0ABDA0',
	}); 
});
cy_graph.nodes('node[name = "Concat"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#4DD8AD',
		'text-outline-color': '#4DD8AD',
	}); 
});
cy_graph.nodes('node[name = "Exp"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#888C46',
		'text-outline-color': '#888C46',
	}); 
});
cy_graph.nodes('node[name = "Log"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#93A806',
		'text-outline-color': '#93A806',
	}); 
});
cy_graph.nodes('node[name = "Power"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#BED905',
		'text-outline-color': '#BED905',
	}); 
});
cy_graph.nodes('node[name = "Softmax"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#D6618F',
		'text-outline-color': '#D6618F',
	}); 
});
cy_graph.nodes('node[name = "Activation"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#F3D4A0',
		'text-outline-color': '#F3D4A0',
	}); 
});
cy_graph.nodes('node[name = "ReLU"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#53C0F0',
		'text-outline-color': '#53C0F0',
	}); 
});
cy_graph.nodes('node[name = "PReLU"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#8F715B',
		'text-outline-color': '#8F715B',
	}); 
});
cy_graph.nodes('node[name = "ELU"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#270101',
		'text-outline-color': '#270101',
	}); 
});
cy_graph.nodes('node[name = "Dense"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#F1931B',
		'text-outline-color': '#F1931B',
	}); 
});
cy_graph.nodes('node[name = "Dropout"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#0294A5',
		'text-outline-color': '#0294A5',
	}); 
});
cy_graph.nodes('node[name = "Flatten"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#BBBF95',
		'text-outline-color': '#BBBF95',
	}); 
});
cy_graph.nodes('node[name = "Permute"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#54678F',
		'text-outline-color': '#54678F',
	}); 
});
cy_graph.nodes('node[name = "Cropping"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#8D2F23',
		'text-outline-color': '#8D2F23',
	}); 
});
cy_graph.nodes('node[name = "Slice"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#A3765D',
		'text-outline-color': '#A3765D',
	}); 
});
cy_graph.nodes('node[name = "BatchNorm"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#F3B05A',
		'text-outline-color': '#F3B05A',
	}); 
});
cy_graph.nodes('node[name = "LRN"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#BE9063',
		'text-outline-color': '#BE9063',
	}); 
});
cy_graph.nodes('node[name = "MVN"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#A4978E',
		'text-outline-color': '#A4978E',
	}); 
});
cy_graph.nodes('node[name = "Pooling"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#F4874B',
		'text-outline-color': '#F4874B',
	}); 
});
cy_graph.nodes('node[name = "SPP"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#0444BF',
		'text-outline-color': '#0444BF',
	}); 
});
cy_graph.nodes('node[name = "Convolution"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#F46A4E',
		'text-outline-color': '#F46A4E',
	}); 
});
cy_graph.nodes('node[name = "DeSepConvolution"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#C05640',
		'text-outline-color': '#C05640',
	}); 
});
cy_graph.nodes('node[name = "Deconvolution"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#EDD170',
		'text-outline-color': '#EDD170',
	}); 
});
cy_graph.nodes('node[name = "RNN"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#BD3E85',
		'text-outline-color': '#BD3E85',
	}); 
});
cy_graph.nodes('node[name = "Embedding"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#7B8937',
		'text-outline-color': '#7B8937',
	}); 
});
cy_graph.nodes('node[name = "Scale"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#85144b',
		'text-outline-color': '#85144b',
	}); 
});
cy_graph.nodes('node[name = "Reshape"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#01FF70',
		'text-outline-color': '#01FF70',
	}); 
});
cy_graph.nodes('node[name = "Split"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#360CAE',
		'text-outline-color': '#350CAE',
	}); 
});
cy_graph.nodes('node[name = "Output"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#4BBFCA',
		'text-outline-color': '#4BBFCA',
	}); 
});
cy_graph.nodes('node[name = "DeformConvolution"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#CB592F',
		'text-outline-color': '#CB592F',
	}); 
});
cy_graph.nodes('node[name = "Deconvolution"]').forEach(function( ele ){
	ele.style({ 
		'background-color': '#EF9C7D',
		'text-outline-color': '#EF9C7D',
	}); 
});
