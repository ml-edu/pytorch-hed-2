Holistically-Nested Edge Detection: ``pytorch-hed`` 
==========================================================

.. raw:: html

   <blockquote>
   <p>This is a reimplementation in the form of a python package of  
   <a href="#references">Holistically-Nested Edge Detection</a> using PyTorch based on 
   the previous pytorch implementation by <a href="https://github.com/sniklaus">sniklaus</a>. 
   If you would like to use of this work, please cite the paper accordingly. 
   Also, make sure to adhere to the licensing terms of the authors. Moreover, if you 
   will be making use of <a href="https://github.com/Davidelanz/pytorch-hed#references">
   this particular implementation</a>, please acknowledge it.</p>
   </blockquote>

   <p><a href="https://arxiv.org/abs/1504.06375" rel="nofollow"><img src="https://camo.githubusercontent.com/f78dbb429ab1e69d87cd863a7a73f0ab89c05c3712dc2763491295c867170a23/687474703a2f2f7777772e61727869762d73616e6974792e636f6d2f7374617469632f7468756d62732f313530342e303633373576322e7064662e6a7067" alt="Paper" width="100%" data-canonical-src="http://www.arxiv-sanity.com/static/thumbs/1504.06375v2.pdf.jpg" style="max-width:100%;"></a></p>

|


-----------------
Usage
-----------------

.. code :: python

   import torchHED
   
   # process a single image file 
   torchHED.process_file("./images/sample.png", "./images/sample_processed.png")
   
   # process all images in a folder
   torchHED.process_folder("./input_folder", "./output_folder")

   # process a PIL.Image loaded in memory and return a new PIL.Image
   # img = PIL.Image.open("./images/sample.png")
   img_hed = torchHED.process_img(img)


.. raw:: html
   
   <table>
      <thead>
         <tr>
            <th>Input</th>
            <th>Output</th>
         </tr>
      </thead>
      <tbody>
         <tr>
            <td><a target="_blank" rel="noopener noreferrer" href="https://github.com/Davidelanz/pytorch-hed/blob/master/images/sample.png?raw=true"><img src="https://github.com/Davidelanz/pytorch-hed/raw/master/images/sample.png?raw=true" alt="sample" style="max-width:100%;"></a></td>
            <td><a target="_blank" rel="noopener noreferrer" href="https://github.com/Davidelanz/pytorch-hed/blob/master/images/torchHED.png?raw=true"><img src="https://github.com/Davidelanz/pytorch-hed/raw/master/images/torchHED.png?raw=true" alt="sample" style="max-width:100%;"></a></td>
         </tr>
      </tbody>
   </table>


|


-----------------
Documentation
-----------------



.. automodule:: torchHED.hed
   :members:


