//
//  mnistCNN.swift
//  iOS-MPSCNN-MNIST
//
//  Created by Sri Raghu Malireddi on 05/12/17.
//  Copyright © 2017 Sri Raghu Malireddi. All rights reserved.
//

import MetalPerformanceShaders
import Accelerate

func makeGraph(_ device: MTLDevice) -> MPSNNGraph {
    // DeepMNIST model
    let conv1 = MPSCNNConvolutionNode(source: MPSNNImageNode(handle: nil),
                                      weights: MyWeights("conv1", 5, 5, 1, 32))
    let conv1_relu = MPSCNNNeuronReLUNode(source: conv1.resultImage)
    let pool1 = MPSCNNPoolingMaxNode(source: conv1_relu.resultImage, filterSize: 2)
    let conv2 = MPSCNNConvolutionNode(source: pool1.resultImage,
                                      weights: MyWeights("conv2", 5, 5, 32, 64))
    let conv2_relu = MPSCNNNeuronReLUNode(source: conv2.resultImage)
    let pool2 = MPSCNNPoolingMaxNode(source: conv2_relu.resultImage, filterSize: 2)
    let fc1 = MPSCNNFullyConnectedNode(source: pool2.resultImage,
                                       weights: MyWeights("fc1", 7, 7, 64, 1024))
    let fc2 = MPSCNNFullyConnectedNode(source: fc1.resultImage,
                                       weights: MyWeights("fc2", 1, 1, 1024, 10))
    let softmax = MPSCNNSoftMaxNode(source: fc2.resultImage)
    return MPSNNGraph(device: device, resultImage: softmax.resultImage)!
}

func getLabel(finalLayer: MPSImage) -> UInt {
    // even though we have 10 labels outputed the MTLTexture format used is RGBAFloat16 thus 3 slices will have 3*4 = 12 outputs
    var result_half_array = [UInt16](repeating: 6, count: 12)
    var result_float_array = [Float](repeating: 0.3, count: 10)
    for i in 0...2 {
        finalLayer.texture.getBytes(&(result_half_array[4*i]),
                                    bytesPerRow: MemoryLayout<UInt16>.size*1*4,
                                    bytesPerImage: MemoryLayout<UInt16>.size*1*1*4,
                                    from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                    size: MTLSize(width: 1, height: 1, depth: 1)),
                                    mipmapLevel: 0,
                                    slice: i)
    }
    
    // we use vImage to convert our data to float16, Metal GPUs use float16 and swift float is 32-bit
    var fullResultVImagebuf = vImage_Buffer(data: &result_float_array, height: 1, width: 10, rowBytes: 10*4)
    var halfResultVImagebuf = vImage_Buffer(data: &result_half_array , height: 1, width: 10, rowBytes: 10*2)
    
    if vImageConvert_Planar16FtoPlanarF(&halfResultVImagebuf, &fullResultVImagebuf, 0) != kvImageNoError {
        print("Error in vImage")
    }
    
    // poll all labels for probability and choose the one with max probability to return
    var max:Float = 0
    var mostProbableDigit = 10
    
    for i in 0...9 {
        if(max < result_float_array[i]){
            max = result_float_array[i]
            mostProbableDigit = i
        }
    }
    
    return UInt(mostProbableDigit)
}
