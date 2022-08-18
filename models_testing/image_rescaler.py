import torch

def increase_img_resolution(img_tensor, resolution_model, res_mod = 2, split_img = True):
    #Split img in to chunks
    if split_img:
        part1, part2 = img_tensor.chunk(2, axis=2)
        chunk1, chunk2  = part1.chunk(2, axis=3)
        chunk3, chunk4  = part2.chunk(2, axis=3)
        if res_mod <= 2:
            batch = torch.concat([chunk1, chunk2, chunk3, chunk4])
        else:
            chunks = []
            for chunk in [chunk1, chunk2, chunk3, chunk4]:
                res_cunk = increase_img_resolution(chunk, resolution_model, res_mod = res_mod // 2)
                chunks.append(res_cunk)
            batch = torch.concat(chunks)
    else:
        batch = img_tensor

    with torch.no_grad():
        #give chunks into model
        output, _ = resolution_model(batch)

        #concat outputs
        if split_img:
            out1, out2, out3, out4 = output.chunk(4, axis=0)
            out_part1 = torch.cat((out1, out2), 3)
            out_part2 = torch.cat((out3, out4), 3)
            rev_out = torch.cat((out_part1, out_part2), 2)
        else:
            rev_out = output
        return rev_out