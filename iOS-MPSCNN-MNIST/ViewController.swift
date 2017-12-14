//
//  ViewController.swift
//  iOS-MPSCNN-MNIST
//
//  Created by Sri Raghu Malireddi on 05/12/17.
//  Copyright © 2017 Sri Raghu Malireddi. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

class ViewController: UIViewController {
    
    @IBOutlet weak var drawView: DrawView!
    @IBOutlet weak var predictLabel: UILabel!
    
    // Params: Metal Performance Shaders
    var device: MTLDevice!
    var graph: MPSNNGraph!
    var inputImage: MPSImage!
    var inputImageDesc: MPSImageDescriptor!
    
    // MNIST dataset image parameters
    let mnistInputWidth  = 28
    let mnistInputHeight = 28
    let mnistInputNumPixels = 784

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        predictLabel.isHidden = true
        
        device = MTLCreateSystemDefaultDevice()
        inputImageDesc = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.unorm8,
                                            width: mnistInputWidth,
                                            height: mnistInputHeight,
                                            featureChannels: 1)
        inputImage = MPSImage(device: device, imageDescriptor: inputImageDesc)
        graph = makeGraph(device)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func tappedClear(_ sender: Any) {
        drawView.lines = []
        drawView.setNeedsDisplay()
        predictLabel.isHidden = true
    }
    
    @IBAction func tappedDetect(_ sender: Any) {
        let context = drawView.getViewContext()
        inputImage.texture.replace(region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z:0),
                                                     size: MTLSize(width: mnistInputWidth,
                                                                   height: mnistInputHeight,
                                                                   depth: 1)),
                                   mipmapLevel: 0,
                                   slice: 0,
                                   withBytes: context!.data!,
                                   bytesPerRow: mnistInputWidth,
                                   bytesPerImage: 0)
        
        graph?.executeAsync(withSourceImages: [inputImage]) {
            resultImage, error in
            // check for error and use resultImage inside closure
            if let image = resultImage {
                DispatchQueue.main.async {
                    self.predictLabel.text = String(getLabel(finalLayer: image))
                    self.predictLabel.isHidden = false
                }
            }
            else {
                print("Error!: not detected")
            }
        }
    }
    
}

