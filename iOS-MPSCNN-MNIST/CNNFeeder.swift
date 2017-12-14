//
//  CNNFeeder.swift
//  iOS-MPSCNN-MNIST
//
//  Created by Sri Raghu Malireddi on 05/12/17.
//  Copyright © 2017 Sri Raghu Malireddi. All rights reserved.
//

import MetalPerformanceShaders

class MyWeights: NSObject, MPSCNNConvolutionDataSource {
    
    // Some variables to initialize
    let name: String
    let kernelWidth, kernelHeight, inputFeatureChannels, outputFeatureChannels: Int
    
    var W: UnsafeMutableRawPointer?
    var b: UnsafeMutablePointer<Float>?
    
    // Initialize the data source object
    init(_ name: String, _ kernelWidth: Int, _ kernelHeight: Int,
         _ inputFeatureChannels: Int, _ outputFeatureChannels: Int) {
        self.name = name
        self.kernelWidth = kernelWidth
        self.kernelHeight = kernelHeight
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
    }
    
    public func load() -> Bool {
        // Function to load weights and bias into convolution descriptor
        
        // calculate the size of weights and bias required to be memory mapped into memory
        let sizeBias = outputFeatureChannels * MemoryLayout<Float>.size
        let sizeWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels * MemoryLayout<Float>.size
        
        // get the url to this layer's weights and bias
        let wtPath = Bundle.main.path(forResource: "weights_" + name, ofType: "dat")
        let bsPath = Bundle.main.path(forResource: "bias_" + name, ofType: "dat")
        
        // open file descriptors in read-only mode to parameter files
        let fd_w = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        let fd_b = open( bsPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        
        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")
        assert(fd_b != -1, "Error: failed to open output file at \""+bsPath!+"\"  errno = \(errno)\n")
        
        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0)
        let hdrB = mmap(nil, Int(sizeBias), PROT_READ, MAP_FILE | MAP_SHARED, fd_b, 0)
        
        // cast Void pointers to Float
        W = UnsafeMutableRawPointer(hdrW!.bindMemory(to: Float.self, capacity: sizeWeights))
        b = UnsafeMutablePointer(hdrB!.bindMemory(to: Float.self, capacity: sizeBias))
        
        // close file descriptors
        close(fd_w)
        close(fd_b)
        
        return true
    }
    
    public func descriptor() -> MPSCNNConvolutionDescriptor {
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                               kernelHeight: kernelHeight,
                                               inputFeatureChannels: inputFeatureChannels,
                                               outputFeatureChannels: outputFeatureChannels)
        return desc
    }
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func label() -> String? {
        return name
    }
    
    public func weights() -> UnsafeMutableRawPointer {
        return W!
    }
    
    public func biasTerms() -> UnsafeMutablePointer<Float>? {
        return b
    }
    
    public func purge() {
        W = nil
        b = nil
    }
}
