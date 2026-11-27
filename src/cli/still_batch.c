#include "src/cli/still_batch.h"
#include "src/internal.h"
#include "src/execution.h"
#include "src/vae/image_vae.h"
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *error,size_t size,const char *message) {
    if(error&&size)snprintf(error,size,"%s",message);
    return 0;
}

char *h3_still_batch_path(const char *path,int count,uint64_t seed) {
    if(!path)return NULL;
    if(count==1)return strdup(path);
    const char *base=strrchr(path,'/');base=base?base+1:path;
    const char *extension=strrchr(base,'.');
    size_t prefix=extension?(size_t)(extension-path):strlen(path);
    if(prefix>INT_MAX)return NULL;
    char *result=NULL;
    if(asprintf(&result,"%.*s-seed-%" PRIu64 "%s",(int)prefix,path,seed,extension?extension:"")<0)return NULL;
    return result;
}

int h3_still_batch_validate(const h3_params *params,int count,char *error,size_t size) {
    if(!params||!params->still)return fail(error,size,"--still-count requires --still");
    if(count<1)return fail(error,size,"--still-count must be a positive integer");
    if((uint64_t)(count-1)>UINT64_MAX-params->seed)
        return fail(error,size,"--still-count would overflow the initial seed plus N-1");
    return 1;
}

static int item_params(const h3_params *source,int count,int index,h3_params *item,
    char **output,char **latent,char *error,size_t size) {
    *item=*source;item->seed=source->seed+(uint64_t)index;
    *output=h3_still_batch_path(source->output_path,count,item->seed);
    *latent=h3_still_batch_path(source->save_still_latent,count,item->seed);
    if((source->output_path&&!*output)||(source->save_still_latent&&!*latent)) {
        free(*output);free(*latent);*output=*latent=NULL;
        return fail(error,size,"cannot allocate still batch filenames");
    }
    item->output_path=*output;item->save_still_latent=*latent;
    return h3_still_options(item,error,size);
}

int h3_still_batch_run(h3_ctx *ctx,const char *prompt,const h3_params *params,
    int count,const h3_still_batch_hooks *hooks,char *error,size_t size) {
    if(error&&size)*error=0;
    if(!ctx||!prompt)return fail(error,size,"missing still batch context or prompt");
    if(!h3_still_batch_validate(params,count,error,size)||!h3_still_options(params,error,size))return 0;
    FILE *staged=tmpfile();
    if(!staged)return fail(error,size,"cannot create temporary still batch latent storage");
    int ok=0,previous_cache=ctx->cache_enabled,decode_policy_set=0;
    h3_cuda_policy previous_policy={0};
    h3_still_latent shape={0};h3_still_decode_cache decoder={0};
    char *vae=NULL,*output=NULL,*latent_path=NULL,*provenance=NULL;
    h3_result *result=NULL;
    size_t elements=(size_t)24*(size_t)(params->height/16)*(size_t)(params->width/16);
    h3_cache_set_enabled(ctx,1);
    if(asprintf(&vae,"%s/%s/video_vae/source",ctx->model_dir,params->reference_count?"Ref2VA":"FL2VA")<0) {
        fail(error,size,"cannot resolve still batch VAE");goto done;
    }
    for(int index=0;index<count;index++) {
        h3_params item;
        if(!item_params(params,count,index,&item,&output,&latent_path,error,size))goto done;
        if(hooks&&hooks->item)hooks->item(index,count,item.seed,0,hooks->opaque);
        result=h3_generate_still_latent(ctx,prompt,&item);
        if(!result){fail(error,size,h3_last_error(ctx));goto done;}
        const h3_still_latent *z=result->still_latent;
        if(!z||z->height!=params->height/16||z->width!=params->width/16||!z->values) {
            fail(error,size,"still batch returned an incompatible latent");goto done;
        }
        if(!index) {
            shape=*z;shape.values=NULL;
            if(result->lora_provenance) {
                provenance=strdup(result->lora_provenance);
                if(!provenance){fail(error,size,"cannot retain still batch LoRA provenance");goto done;}
            }
        } else if(z->reference_model!=shape.reference_model||memcmp(z->compatibility_sha256,shape.compatibility_sha256,65)) {
            fail(error,size,"still batch VAE identity changed");goto done;
        }
        if(fwrite(z->values,sizeof(float),elements,staged)!=elements) {
            fail(error,size,"cannot stage still batch latents (check temporary disk space)");goto done;
        }
        h3_result_free(result);result=NULL;
        free(output);output=NULL;free(latent_path);latent_path=NULL;
    }
    if(fflush(staged)||fseeko(staged,0,SEEK_SET)) {
        fail(error,size,"cannot rewind still batch latent storage");goto done;
    }
    /* The decoder and prepared transformer must never occupy memory together. */
    h3_cache_clear(ctx);
    /* Preserve the request's CUDA weight/allocation controls during delivery,
     * just as when the VAE runs inside h3_generate for a single still. */
    h3_cuda_policy policy;
    if(!h3_cuda_policy_resolve(params,ctx->device.backend,ctx->lora!=NULL,&policy,error,size) ||
       !h3_cuda_policy_preflight(&policy,error,size))goto done;
    previous_policy=h3_cuda_policy_exchange(policy);decode_policy_set=1;
    for(int index=0;index<count;index++) {
        h3_params item;
        if(!item_params(params,count,index,&item,&output,&latent_path,error,size))goto done;
        if(hooks&&hooks->item)hooks->item(index,count,item.seed,1,hooks->opaque);
        h3_still_latent *z=malloc(sizeof(*z));
        if(!z){fail(error,size,"cannot allocate staged still metadata");goto done;}
        *z=shape;z->values=malloc(elements*sizeof(float));
        if(!z->values||fread(z->values,sizeof(float),elements,staged)!=elements) {
            h3_still_latent_free(z);free(z);
            fail(error,size,"cannot read staged still batch latents");goto done;
        }
        h3_decode_options options={.output_path=output,.on_frame=item.on_frame,
            .on_progress=item.on_progress,.callback_opaque=item.callback_opaque};
        result=h3_decode_still_values_cached(z,vae,&options,&decoder,error,size);
        if(!result){h3_still_latent_free(z);free(z);goto done;}
        result->still_latent=z;result->seed=item.seed;
        result->completed_steps=item.steps;result->total_steps=item.steps;
        if(provenance) {
            result->lora_provenance=strdup(provenance);
            if(!result->lora_provenance){fail(error,size,"cannot retain still result provenance");goto done;}
        }
        if(hooks&&hooks->result)hooks->result(result,output,hooks->opaque);
        h3_result_free(result);result=NULL;
        free(output);output=NULL;free(latent_path);latent_path=NULL;
    }
    ok=1;
done:
    h3_result_free(result);h3_still_decode_cache_clear(&decoder);
    h3_cache_clear(ctx);h3_cache_set_enabled(ctx,previous_cache);
    if(decode_policy_set)h3_cuda_policy_exchange(previous_policy);
    free(vae);free(output);free(latent_path);free(provenance);fclose(staged);
    return ok;
}
