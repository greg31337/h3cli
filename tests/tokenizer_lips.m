/* Coarse mouth-motion observations for generated dialogue. This is not a
 * phoneme/lip-sync scorer. Largest detected face, inner/outer lip aspect ratio.
 * Usage: tokenizer_lips frames-directory (PNG frames sampled at 12 fps). */
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static id aspect(VNFaceLandmarkRegion2D *lips, CGRect face) {
    if (!lips || !lips.pointCount) return NSNull.null;
    double x0=DBL_MAX, x1=-DBL_MAX, y0=DBL_MAX, y1=-DBL_MAX;
    for (NSUInteger i=0; i<lips.pointCount; i++) {
        CGPoint p=lips.normalizedPoints[i];
        x0=fmin(x0,p.x); x1=fmax(x1,p.x);
        y0=fmin(y0,p.y); y1=fmax(y1,p.y);
    }
    if (x1<=x0 || face.size.width<=0) return NSNull.null;
    return @((y1-y0)*face.size.height/((x1-x0)*face.size.width));
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        NSString *directory=[NSString stringWithUTF8String:argv[1]];
        NSArray *files=[[NSFileManager.defaultManager contentsOfDirectoryAtPath:directory error:NULL]
                        sortedArrayUsingSelector:@selector(compare:)];
        NSMutableArray *rows=[NSMutableArray array];
        NSUInteger index=0;
        for (NSString *name in files) {
            if (![name.pathExtension isEqual:@"png"]) continue;
            NSURL *url=[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:name]];
            CGImageSourceRef source=CGImageSourceCreateWithURL((__bridge CFURLRef)url,NULL);
            CGImageRef image=source?CGImageSourceCreateImageAtIndex(source,0,NULL):NULL;
            if (!image) return 1;
            VNDetectFaceLandmarksRequest *request=[[VNDetectFaceLandmarksRequest alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            request.usesCPUOnly=YES;
#pragma clang diagnostic pop
            VNImageRequestHandler *handler=[[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
            NSError *error=nil;
            if (![handler performRequests:@[request] error:&error]) {
                fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); return 1;
            }
            VNFaceObservation *face=nil;
            for (VNFaceObservation *candidate in request.results)
                if (!face || candidate.boundingBox.size.width*candidate.boundingBox.size.height>
                             face.boundingBox.size.width*face.boundingBox.size.height) face=candidate;
            NSMutableDictionary *row=[@{@"time":@((double)index++/12),@"frame":name} mutableCopy];
            if (face) {
                CGRect box=face.boundingBox;
                row[@"face_box"]=@[@(box.origin.x),@(box.origin.y),@(box.size.width),@(box.size.height)];
                row[@"inner_lip_aspect"]=aspect(face.landmarks.innerLips,box);
                row[@"outer_lip_aspect"]=aspect(face.landmarks.outerLips,box);
                row[@"confidence"]=@(face.confidence);
            }
            [rows addObject:row];
            CGImageRelease(image); CFRelease(source);
        }
        NSData *json=[NSJSONSerialization dataWithJSONObject:rows options:NSJSONWritingPrettyPrinted error:NULL];
        fwrite(json.bytes,1,json.length,stdout); putchar('\n');
    }
    return 0;
}
