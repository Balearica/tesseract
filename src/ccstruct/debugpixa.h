#ifndef TESSERACT_CCSTRUCT_DEBUGPIXA_H_
#define TESSERACT_CCSTRUCT_DEBUGPIXA_H_

#include "image.h"

#include <allheaders.h>

#include "scrollview.h"

namespace tesseract {

// Class to hold a Pixa collection of debug images with captions and save them
// to a PDF file.
class DebugPixa {
public:
  // TODO(rays) add another constructor with size control.
  DebugPixa() {
    pixa_ = pixaCreate(0);
#ifdef TESSERACT_DISABLE_DEBUG_FONTS
    fonts_ = NULL;
#else
    fonts_ = bmfCreate(nullptr, 14);
#endif
  }
  // If the filename_ has been set and there are any debug images, they are
  // written to the set filename_.
  ~DebugPixa() {
    pixaDestroy(&pixa_);
    bmfDestroy(&fonts_);
  }

  // Adds the given pix to the set of pages in the PDF file, with the given
  // caption added to the top.
  void AddPix(const Image pix, const char *caption) {

  #ifdef IMAGEMASK_SCROLLVIEW
  #ifndef GRAPHICS_DISABLED
    if (pix != nullptr)  {

      // Some uses of this function dump potentially hundreds of individual images, which do not align with the entire page.
      // These are skipped when using the scrollview version of this function.
      if (strncmp(caption, "A component", strlen("A component")) == 0 ||
        strncmp(caption, "ImageComponent", strlen("ImageComponent")) == 0) {
        return;
      }

      int width = pixGetWidth(pix);
      int height = pixGetHeight(pix);
      auto *win = new ScrollView(caption, 0, 0, width, height, width, height);

      if (pix != nullptr) {
        win->Draw(pix, 0, 0);
      }

      win->UpdateWindow();
      #ifdef SCROLLVIEW_NONINTERACTIVE
      delete win;
      #endif
    }
  #endif
  #else
    int depth = pixGetDepth(pix);
    int color = depth < 8 ? 1 : (depth > 8 ? 0x00ff0000 : 0x80);
    Image pix_debug =
        pixAddSingleTextblock(pix, fonts_, caption, color, L_ADD_BELOW, nullptr);
    pixaAddPix(pixa_, pix_debug, L_INSERT);
  #endif
  }

  // Sets the destination filename and enables images to be written to a PDF
  // on destruction.
  void WritePDF(const char *filename) {
    if (pixaGetCount(pixa_) > 0) {
      pixaConvertToPdf(pixa_, 300, 1.0f, 0, 0, "AllDebugImages", filename);
      pixaClear(pixa_);
    }
  }

private:
  // The collection of images to put in the PDF.
  Pixa *pixa_;
  // The fonts used to draw text captions.
  L_Bmf *fonts_;
};

} // namespace tesseract

#endif // TESSERACT_CCSTRUCT_DEBUGPIXA_H_
