macro "Save Mask as TIF [A]" {
    // ===== CONFIGURE THIS =====
    outputDir = "Z:\\Users\\Artin\\coiled\\masks_both_view\\";  // Windows path, include trailing slash
    // ==========================

    // Get the title of the current image (including extension)
    origTitle = getTitle();

    // Strip off the extension
    dot = lastIndexOf(origTitle, ".");
    if (dot > 0)
        base = substring(origTitle, 0, dot);
    else
        base = origTitle;

    // If it starts with "AVG_", drop those first 4 characters
    if (startsWith(base, "AVG_"))
        base = substring(base, 4);

    // Build the output path with .tif
    outPath = outputDir + base + ".tif";

    // Create a binary mask from the current ROI
    run("Create Mask");

    // Save it
    saveAs("Tiff", outPath);

    // Close all open images
    run("Close All");
}

