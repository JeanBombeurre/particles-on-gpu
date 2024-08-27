"use strict";
/////nouvelle encapsulation de webgpu. la derniere etait trop lourde+trop héritée de webgl
const log=console.log
const f32=d=>new Float32Array(d)
const u32= d=> new Uint32Array(d);
const i32= d=> new Int32Array(d);
function unspace(s){return s.replace(/\s+/g, '')}//enleve espaces
class GlobalsGPU{
	static device;
	static ctx
	static presentationFormat;
	static inPass=false;
	static hasToEnd=false///indique si c'est GlobalsGPU qui doit end le pass actuel
	static initGPU(){//////////////TODO
	}
	static setGlobals(obj){//set les structures internes de Action telles que definies dans obj
		/////utiliser si setupWebGPU() a pas été utilisé, mais que tout a été fait à la main
		let l=["device","ctx","presentationFormat"]
		for(let i of l){
			if(obj[i]===undefined) throw "manque structure "+i
			GlobalsGPU[i]=obj[i];
		}
	}
	static startPass(pass=-1){//commence 
		if(GlobalsGPU.inPass) throw 42
		GlobalsGPU.inPass=true;
		if(pass==-1){
			GlobalsGPU.encoder=GlobalsGPU.device.createCommandEncoder({label: "encoder"})
			GlobalsGPU.pass=GlobalsGPU.encoder.beginComputePass();
			GlobalsGPU.hasToEnd=true
		}else{
			GlobalsGPU.pass=pass;
			GlobalsGPU.hasToEnd=false
		}
	}
	static endPass(){///termine et exécute
		if(!GlobalsGPU.inPass) throw 43
		GlobalsGPU.inPass=false;
		if(GlobalsGPU.hasToEnd){
			GlobalsGPU.pass.end();
			GlobalsGPU.device.queue.submit([GlobalsGPU.encoder.finish()])
		}
	}
}
class Data{
	constructor(type, usage, size,{flags=0,label=undefined, fromBuffer=false} = {}){
	///type : "CANVAS", "vec4f","vec4u", ..., "array<vec4u>"..., "vec4f+vec4u" pour structure (uniform) 
	///si type=="void", pas de type checking...
	///usage: "uniform" ou "storage"
	///size : nombre de bytes. (automatiser ???) 
	////flags : flags GPUBufferStorage.XXX additionnels
		type=unspace(type)
		this.label=label
		this.type=type
		if(type=="CANVAS") return 
		this.usage=usage
		this.size=size
		let d={
			uniform:GPUBufferUsage.COPY_DST|GPUBufferUsage.UNIFORM,
			storage: GPUBufferUsage.STORAGE
		}
		if(!d[usage]) throw "usage ("+usage+") inconnu. usages possibles : [storage, uniform]"
		this.buffer=GlobalsGPU.device.createBuffer({
			size : size,
			usage : d[usage]|flags,
			label:this.label
		})
	}
	static fromBuffer(buf){///buf : un GPUBuffer. sert à intégrer à Action un objet créé à la main
		///dangereux, si je modifie dans le futur le constructor, il faudra modifier ca aussi
		let data=Object.create(this.prototype)
		data.label="external"
		data.type="void"///type générique, pas de type checking
		data.size=buf.size
		data.buffer=buf;
		return data
	}
	getResource(){
		if(this.type=="CANVAS") return GlobalsGPU.ctx.getCurrentTexture().createView()
		return {buffer :this.buffer }
	}
	write(data){//seulement pour "uniform"
		GlobalsGPU.device.queue.writeBuffer(this.buffer, 0,data);
	}
}

class Action{///////////////NODE DE CALCUL.
	static i=0;
	constructor(str,name="", type="compute"){//autres types pas encore gérés
		const d=GlobalsGPU.device;
		this.str=str;
		if(name==="") name="unnamed "+Action.i++;
		this.name=name
		this.preprocess()
		this.module=d.createShaderModule({label:`[${this.name}] module`, code:this.code})
		
		if(type!="compute") throw "shader type pas codé"
		this.pipeline=d.createComputePipeline({
			label:`[${name}] pipeline`,
			layout:"auto", //////////////////////////////////////paramétrer ?...
			compute:{module:this.module},
			constants : Action.constants,
		})
	}
	static defines(str){//déclaration ajoutée à chaque début de shader
		Action.defs=str
	}
	preprocess(){///ajoute fonctions utiles au debut, et surtout repère les arguments, leurs types
		//replaces
		let inu=0;
		this.code=this.str.replace("$CANVAS", "texture_storage_2d<"+GlobalsGPU.presentationFormat+",write>");
		this.code=this.code.replaceAll("#", (...a) => `@group(0) @binding(${inu++}) `)
		this.code=this.code.replaceAll("$ID","@builtin(global_invocation_id) id : vec3u")
		this.code=Action.defs?Action.defs+'\n'+this.code:this.code
		
		this.bindingList=[]//liste d'indices
		let args={}
		let pattern=/@group\(0\)\s+@binding\((\d+)\)\s+var[<>\s,\w]*\s+(\w+)\s*:\s*([,<>\w]+)\s*;/g
		for(let x of this.code.matchAll(pattern)){
			let i=parseInt(x[1])//num binding
			let name=x[2]
			let type=x[3].replace(/\s+/g, '')
			args[i]={name:name, type:type}
			if(!this.bindingList.includes(i))this.bindingList.push(i)
			else throw "deux bindings identiques en "+this.name
		}
		this.argsType=args

		let wgs=this.code.matchAll(/@workgroup_size\(([,\s\d]+)\)/g)
		this.wgs=-1;
		for(let i of wgs){
			if(this.wgs!=-1) throw "deux workgroupsizes?"
			this.wgs=i[1]
		}
	}
	makeBindGroup(args){//// args  : {name:data...} comme args
		let entries=[]
		for(let i of this.bindingList){
			let data=args[this.argsType[i].name]

			if(!data) throw "argument missing :<"+this.argsType[i].name+"> in shader : \n\n\n"+this.code
			if(data.type!=this.argsType[i].type && data.type!="CANVAS" && data.type!="void") throw "types pas coherents : "+data.type+" dans "+this.argsType[i].type
			entries.push({binding:i, resource:data.getResource()})	
		}
		let bindGroup=GlobalsGPU.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0), ////   (?)
			entries: entries
		});
		return bindGroup
	}
	apply(bg, nx, ny=1,nz=1){///bg : le bindgroup actuel
		///bg = -1 : utiliser this.bg
		//TODO ajouter option pour ne pas reset pipeline et bg
		if(!GlobalsGPU.inPass) throw "pass?"
		if(bg==-1) bg=this.bg
		GlobalsGPU.pass.setPipeline(this.pipeline)
		GlobalsGPU.pass.setBindGroup(0,bg)
		GlobalsGPU.pass.dispatchWorkgroups(nx,ny,nz)
	}
	fixBindGroup(args){//attribue un bg fixé
		let bg=this.makeBindGroup(args)
		this.hasFixedBindGroup=1
		this.bg=bg
	}
}

async function read(buffer,type=f32){//lit array<u32>
	let device=GlobalsGPU.device
	const encoder = device.createCommandEncoder();
	let staging=device.createBuffer({mappedAtCreation:false, size:buffer.size, usage : GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})
	encoder.copyBufferToBuffer(
		buffer,//src
		0,
		staging,//dst
		0,
		buffer.size,
	);
	device.queue.submit([encoder.finish()]);

	await staging.mapAsync(
		GPUMapMode.READ,
		0,
		staging.byteLength,
	).then(()=>{
		const copyArrayBuffer = staging.getMappedRange(0, staging.byteLength);
		const data = copyArrayBuffer.slice(0);
		staging.unmap();

		let inu=type(data)
		console.log(inu);
		
	}) 
}

async function setupWebGPU(ctx){
	const adapter = await navigator.gpu?.requestAdapter();
	const hasBGRA8unormStorage = adapter.features.has('bgra8unorm-storage');
	const device = await adapter?.requestDevice({///POUR RENDER SUR CANVAS
		requiredFeatures: hasBGRA8unormStorage
			? ['bgra8unorm-storage']
			: [],
	});
	GlobalsGPU.device=device
	GlobalsGPU.ctx=ctx
	
	if (!device) {
		fail('need a browser that supports WebGPU');
		return;
	}
	const presentationFormat = hasBGRA8unormStorage
		? navigator.gpu.getPreferredCanvasFormat()
		: 'rgba8unorm';
	ctx.configure({
		device,
		format: presentationFormat,
		usage: GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING,
	});
	GlobalsGPU.presentationFormat=presentationFormat
}